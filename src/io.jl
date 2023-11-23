export read_json_dataframe, read_json_lines, load_emb

function read_json_lines(f, numlines::Integer)
    L = Dict{String, Any}[]

    for line in eachline(f)
        push!(L, JSON.parse(line))
        length(L) == numlines && break
    end

    L
end

function read_json_dataframe(filename; numlines::Int=typemax(Int))
    L = if endswith(filename, ".gz")
        open(GzipDecompressorStream, filename) do f
            read_json_lines(f, numlines)
        end
    else
        open(filename) do f
            read_json_lines(f, numlines)
        end
    end

    DataFrame(L)
end

function read_synonyms(filename)
    Dict(k => first(v) for (k, v) in open(JSON.parse, filename))
end

# loads a h5 file with the normalized embeddings
function load_emb(embfile)
    embeddings, vocab = jldopen(embfile) do f
        f["embeddings"], f["vocab"]
    end

    dist = NormalizedCosineDistance()

    (; vocab, embeddings, dist)
end

