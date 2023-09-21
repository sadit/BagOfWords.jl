export Synonyms, IgnoreStopwords, vocabmap, create_vocabmap, vocab 

struct Synonyms <: AbstractTokenTransformation
    map::Dict{String,String}
    
    Synonyms(mapfile) = new(open(JSON.parse, mapfile))
end

function TextSearch.transform_unigram(tt::Synonyms, tok)
    get(tt.map, tok, tok)
end

struct IgnoreStopwords <: AbstractTokenTransformation
    stopwords::Set{String}
end

function TextSearch.transform_unigram(tt::IgnoreStopwords, tok)
    tok in tt.stopwords ? nothing : tok
end

struct ChainTransformation <: AbstractTokenTransformation
    list::AbstractVector{<:AbstractTokenTransformation}    
end

function TextSearch.transform_unigram(tt::ChainTransformation, tok)
    for t in tt.list
        tok = TextSearch.transform_unigram(t, tok)
        tok === nothing && return nothing
    end 

    tok
end

function vocab(
            text,
            tt=IdentityTokenTransformation();
            nlist=[1], qlist=[], collocations=0, mindocs=3, maxndocs=1.0, 
            textconfig=TextConfig(; nlist, del_punc=false, del_diac=true, lc=true)
    )
    
    V = Vocabulary(TextConfig(textconfig; qlist, collocations, tt), text)

    filter_tokens(V) do t
        mindocs <= t.ndocs < trainsize(V) * maxndocs
    end
end

"""
    create_index(dist::SemiMetric, db::AbstractDatabase; k::Int=16, minrecall::Float64=0.95, verbose=true)

Creates an index for the given database

- `dist`: Distance function
- `db`: input database
- `k`: the number of neighbors (only for optimization purposes)
- `minrecall`: controls the quality of the approximation (between 0 and 1)
- `verbose`: set `verbose=false` to reduce the output of the index's building
"""
function create_index(dist::SemiMetric, db::AbstractDatabase; k::Int=16, minrecall::Float64=0.95, verbose::Bool=true)
    G = SearchGraph(; dist, db, verbose)
    minrecall = MinRecall(minrecall)
    callbacks = SearchGraphCallbacks(minrecall; ksearch=k, verbose)
    index!(G; callbacks)
    optimize!(G, minrecall)
    G
end

function simvocab(names, vocab, embeddings, dist; verbose=true, k=12, minrecall=0.9)
    P = Dict(w => i for (i, w) in enumerate(vocab))
    ivocabidx = Int32[]
    for w in names
        i = get(P, w, 0)
        i > 0 && push!(ivocabidx, i)
    end

    X = StrideMatrixDatabase(embeddings)
    Y = StrideMatrixDatabase(embeddings[:, ivocabidx])
    G = create_index(dist, Y; k, minrecall, verbose)
    
    n = length(vocab)
    knns, dists = searchbatch(G, X, k)
    ivocab = vocab[ivocabidx]

    (; G, vocab, ivocab, ivocabidx, knns, dists)
end

function distquantiles(dist::SemiMetric, X::AbstractDatabase, q=[0.0, 0.25, 0.5, 0.75, 1.0]; samplesize=2^20)
    n = length(X)
    S = Vector{Float32}(undef, samplesize)
    
    Threads.@threads for i in 1:samplesize
        S[i] = evaluate(dist, X[rand(1:n)], X[rand(1:n)])
    end

    quantile(S, q)
end

function vocabmap_(p::NamedTuple, dmax)
    map = Dict{String,String}()
    
    for i in 1:size(p.knns, 2)
       nn = p.knns[1, i]
       dist = p.dists[1, i]
       if dist <= dmax
           map[p.vocab[i]] = p.ivocab[nn]
       end
    end

    map
end

function vocabmap(; outfile::String, quant::AbstractFloat, emb::NamedTuple, train::DataFrame, mindocs::Integer, maxndocs=0.5)
    voc = vocab(train.text; nlist=[1], qlist=[], collocations=0, mindocs, maxndocs)
    s = simvocab(token(voc), emb.vocab, emb.embeddings, emb.dist)
    dmax = distquantiles(emb.dist, StrideMatrixDatabase(emb.embeddings), quant)
    map = vocabmap_(s, dmax)

    open(outfile, "w") do f
        print(f, json(map, 2))
    end

    map
end

function create_vocabmap(config; embfile=embedding_by_lang(config.lang), nick=config.nick, quantlist=[0.01, 0.03, 0.1, 0.3, 1], path="mapping")
    train = read_json_dataframe(config.trainfile)
    emb = load_emb(embfile)
    mkpath(path)

    for mindocs in [1], quant in quantlist
        outfile = joinpath(path, "map-$nick-$quant-$mindocs.json")
        @info "creating $outfile"
        vocabmap(; outfile, quant, mindocs, train, emb)
    end
end
