export read_json_dataframe, load_emb

read_json_lines_(f) = [JSON.parse(line) for line in eachline(f)]

function read_json_dataframe(filename)
    L = if endswith(filename, ".gz")
        open(read_json_lines_, GzipDecompressorStream, filename)
    else
        open(read_json_lines_, filename)
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

