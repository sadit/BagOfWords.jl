module BagOfWords

using TextSearch, SimilaritySearch, DataFrames, Random
using JSON, CodecZlib, JLD2, DataFrames
using LIBSVM, KNearestCenters
using MLUtils, StatsBase
import StatsAPI: predict, fit

export fit, predict, runconfig, BagOfWordsClassifier

include("io.jl")
include("vocab.jl")

struct BagOfWordsClassifier{VectorModel,CLS}
    model::VectorModel
    cls::CLS
end

vectormodel(gw::EntropyWeighting, lw, corpus, labels, V) = VectorModel(gw, lw, V, corpus, labels; comb=SigmoidPenalizeFewSamples())
vectormodel(gw, lw, corpus, labels, V) = VectorModel(gw, lw, V)

function fit(::Type{BagOfWordsClassifier}, corpus, labels, tt=IdentityTokenTransformation();
        gw=EntropyWeighting(),
        lw=BinaryLocalWeighting(),
        collocations::Integer=0,
        nlist::Vector=[1],
        textconfig=TextConfig(; nlist, del_punc=false, del_diac=true, lc=true),
        qlist::Vector=[2, 3],
        mindocs::Integer=1,
        minweight::AbstractFloat=1e-3,
        maxndocs::AbstractFloat=1.0,
        weights=:balanced,
        nt=Threads.nthreads(),
        verbose=false,
        spelling=nothing
    )

    V = let V = vocab(corpus, tt; collocations, nlist, qlist, mindocs, maxndocs)
        spelling === nothing ? V : approxvoc(QgramsLookup, V, DiceDistance(); spelling...)
    end
    model = vectormodel(gw, lw, corpus, labels, V)
    model = filter_tokens(model) do t
        minweight <= t.weight
    end
    X, y, dim = vectorize_corpus(model, corpus), labels, vocsize(model)
    if weights === :balanced
        weights = let C = countmap(y)
            s = sum(values(C))
            nc = length(C)
            Dict(label => (s / (nc * count)) for (label, count) in C)
        end
    end

    BagOfWordsClassifier(model, svmtrain(sparse(X, dim), y; weights, nt, verbose, kernel=Kernel.Linear))
end

function fit(::Type{BagOfWordsClassifier}, corpus, labels, config::NamedTuple)
    tt = config.mapfile === nothing ? IdentityTokenTransformation() : Synonyms(config.mapfile)
    fit(BagOfWordsClassifier, corpus, labels, tt; config.collocations, config.mindocs, config.maxndocs, config.qlist, config.gw, config.lw, config.spelling)
end

function predict(B::BagOfWordsClassifier, corpus; nt=Threads.nthreads())
    Xtest = vectorize_corpus(B.model, corpus)
    dim = vocsize(B.model)
    pred, decision_value = svmpredict(B.cls, sparse(Xtest, dim); nt)
    (; pred, decision_value)
end

function runconfig_(
        config;
        train = read_json_dataframe(config.trainfile),
        test = read_json_dataframe(config.testfile)
    )
    fit(BagOfWordsClassifier)
end

function runconfig(
        config;
        train = read_json_dataframe(config.trainfile),
        test = read_json_dataframe(config.testfile)
    )
    C = fit(BagOfWordsClassifier, train.text, train.klass, config)
    y = predict(C, test.text)
    scores = classification_scores(test.klass, y.pred)
    @info json(scores, 2)

    (; config, scores,
       size=(voc=vocsize(C.model), train=size(train, 1), test=length(y.pred)),
       dist=(train=countmap(train.klass), test=countmap(test.klass), pred=countmap(y.pred))
    )
end


include("modelselection.jl")

end
