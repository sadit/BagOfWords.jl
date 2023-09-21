module BagOfWords

using TextSearch, SimilaritySearch, DataFrames
using JSON, CodecZlib, JLD2, DataFrames
using LIBLINEAR, KNearestCenters
using MLUtils, StatsBase
import StatsAPI: predict, fit

export fit, predict, runconfig

include("io.jl")
include("vocab.jl")


struct BagOfWordsClassier{VectorModel,CLS}
    model::VectorModel
    cls::CLS
end

vectormodel(gw::EntropyWeighting, lw, corpus, labels, V) = VectorModel(gw, lw, V, corpus, labels)
vectormodel(gw, lw, corpus, labels, V) = VectorModel(gw, lw, V)

function fit(::Type{BagOfWordsClassier}, corpus, labels, tt=IdentityTokenTransformation();
        gw=EntropyWeighting(),
        lw=BinaryLocalWeighting(),
        collocations::Integer=0,
        nlist::Vector=[1],
        textconfig=TextConfig(; nlist, del_punc=false, del_diac=true, lc=true),
        qlist::Vector=[2, 3],
        mindocs::Integer=3,
        maxndocs::AbstractFloat=0.5
    )

    V = vocab(corpus, tt; collocations, nlist, qlist, mindocs, maxndocs)
    model = vectormodel(gw, lw, corpus, labels, V)
    X, y, dim = vectorize_corpus(model, corpus), labels, vocsize(model)
    BagOfWordsClassier(model, linear_train(y, sparse(X, dim)))
end

function predict(B::BagOfWordsClassier, corpus)
    Xtest = vectorize_corpus(B.model, corpus)
    dim = vocsize(B.model)
    pred, decision_value = linear_predict(B.cls, sparse(Xtest, dim))
    (; pred, decision_value)
end

function runconfig(
        config;
        train = read_json_dataframe(config.trainfile),
        test = read_json_dataframe(config.testfile),
    )

    tt = config.mapfile === nothing ? IdentityTokenTransformation() : Synonyms(config.mapfile)
    C = fit(BagOfWordsClassier, train.text, train.klass, tt; config.collocations, config.mindocs, config.qlist, config.gw, config.lw)
    y = predict(C, test.text)
    scores = classification_scores(test.klass, y.pred)
    @info json(scores, 2)

    (; config, scores,
       size=(voc=vocsize(C.model), train=size(train, 1), test=length(y.pred)),
       dist=(train=countmap(train.klass), test=countmap(test.klass), pred=countmap(y.pred))
    )
end


end
