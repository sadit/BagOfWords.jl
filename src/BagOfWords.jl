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

#comb=SigmoidPenalizeFewSamples()
vectormodel(gw::EntropyWeighting, lw, corpus, labels, V; smooth=5, comb=NormalizedEntropy()) = VectorModel(gw, lw, V, corpus, labels; comb, smooth)
#vectormodel(gw::EntropyWeighting, lw, corpus, labels, V; smooth=0, comb=SigmoidPenalizeFewSamples()) = VectorModel(gw, lw, V, corpus, labels; comb, smooth)

vectormodel(gw, lw, corpus, labels, V; kwargs...) = VectorModel(gw, lw, V)

function fit(::Type{BagOfWordsClassifier}, corpus, labels, tt=IdentityTokenTransformation();
        gw=EntropyWeighting(),
        lw=BinaryLocalWeighting(),
        collocations::Integer=0,
        nlist::Vector=[1],
        textconfig=TextConfig(; nlist, del_punc=false, del_diac=true, lc=true),
        qlist::Vector=[2, 3],
        mindocs::Integer=3,
        minweight::AbstractFloat=1e-4,
        maxndocs::AbstractFloat=1.0,
        smooth::Real=5,
        comb=NormalizedEntropy(), #SigmoidPenalizeFewSamples(), #NormalizedEntropy(),
        weights=:balanced,
        nt=Threads.nthreads(),
        verbose=false,
        spelling=nothing
    )

    V = let V = vocab(corpus, tt; collocations, nlist, qlist, mindocs, maxndocs)
        spelling === nothing ? V : approxvoc(QgramsLookup, V, DiceDistance(); spelling...)
    end
    model = vectormodel(gw, lw, corpus, labels, V; smooth, comb)
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

    cls = svmtrain(sparse(X, dim), y; weights, nt, verbose, kernel=Kernel.Linear)
    BagOfWordsClassifier(model, cls)
end

function fit(::Type{BagOfWordsClassifier}, corpus, labels, config::NamedTuple)
    tt = config.mapfile === nothing ? IdentityTokenTransformation() : Synonyms(config.mapfile)
    fit(BagOfWordsClassifier, corpus, labels, tt; config.collocations, config.mindocs, config.maxndocs, config.qlist, config.gw, config.lw, config.spelling, config.smooth, config.comb)
end

function predict(B::BagOfWordsClassifier, corpus; nt=Threads.nthreads())
    Xtest = vectorize_corpus(B.model, corpus)
    dim = vocsize(B.model)
    pred, decision_value = svmpredict(B.cls, sparse(Xtest, dim); nt)
    (; pred, decision_value)
end

function runconfig(config, train_text, train_labels, test_text, test_labels)
    C = fit(BagOfWordsClassifier, train_text, train_labels, config)
    y = predict(C, test_text)
    scores = classification_scores(test_labels, y.pred)
    @info json(scores, 2)

    (; config, scores,
       size=(voc=vocsize(C.model), train=length(train_labels), test=length(y.pred)),
       dist=(train=countmap(train_labels), test=countmap(test_labels), pred=countmap(y.pred))
    )
end


include("modelselection.jl")

end
