module BagOfWords

using TextSearch, SimilaritySearch, SimSearchManifoldLearning
using Random, JSON, CodecZlib, JLD2, DataFrames
using LIBLINEAR, LIBSVM
using MLUtils, StatsBase
import StatsAPI: predict, fit

export fit, predict, runconfig, BagOfWordsClassifier,
       classification_scores, f1_score, recall_score, precision_score, accuracy_score,
       RawVectors, UmapProjection, RandomLayout, SpectralLayout,
       NormalizedEntropy, SigmoidPenalizeFewSamples
        

include("io.jl")
include("vocab.jl")

struct BagOfWordsClassifier{VMODEL,PROJ,CLS}
    model::VMODEL
    proj::PROJ
    cls::CLS
end

abstract type SparseProjection end

struct RawVectors <: SparseProjection
    dim::Int32
end

RawVectors() = RawVectors(0)

fit(::RawVectors, X, dim::Integer) = RawVectors(convert(Int32, dim))
predict(proj::RawVectors, X) = sparse(X, proj.dim)

struct UmapProjection{UmapType} <: SparseProjection
    umap::UmapType
    k::Int
    maxoutdim::Int
    n_epochs::Int
    neg_sample_rate::Int
    tol::Float64
    layout
end

function UmapProjection(; k::Integer=15, maxoutdim::Integer=3, n_epochs::Integer=100, layout=SpectralLayout(), neg_sample_rate::Integer=3, tol::AbstractFloat=1e-4)
    UmapProjection(nothing, k, maxoutdim, n_epochs, neg_sample_rate, tol, layout)
end

function fit(U::UmapProjection, X, dim::Integer)
	index = ExhaustiveSearch(; db=VectorDatabase(X), dist=NormalizedCosineDistance())
    umap = fit(UMAP, index; U.k, U.maxoutdim, U.n_epochs, U.neg_sample_rate, U.tol, U.layout)
    UmapProjection(umap, U.k, U.maxoutdim, U.n_epochs, U.neg_sample_rate, U.tol, U.layout)
end

predict(proj::UmapProjection, X) = predict(proj.umap, X)

#comb=SigmoidPenalizeFewSamples()
vectormodel(gw::EntropyWeighting, lw, corpus, labels, V; smooth, comb) = VectorModel(gw, lw, V, corpus, labels; comb, smooth)
vectormodel(gw, lw, corpus, labels, V; kwargs...) = VectorModel(gw, lw, V)

function fit(::Type{BagOfWordsClassifier}, projection::SparseProjection, corpus, labels, tt=IdentityTokenTransformation();
        gw=EntropyWeighting(),
        lw=BinaryLocalWeighting(),
        collocations::Integer=0,
        nlist::Vector=[1],
        textconfig=TextConfig(; nlist, del_punc=false, del_diac=true, lc=true),
        qlist::Vector=[2, 3],
        mindocs::Integer=3,
        minweight::AbstractFloat=1e-4,
        maxndocs::AbstractFloat=1.0,
        smooth::Real=0.5,
        comb=NormalizedEntropy(), #SigmoidPenalizeFewSamples(), #NormalizedEntropy(),
        svm=(; kernel=:linear), # if you use kernel=Kernel.Linear you will be using LIBSVM linear kernel SVM
        spelling=nothing,
        verbose=false
    )

    V = let V = vocab(corpus, tt; collocations, nlist, qlist, mindocs, maxndocs)
        spelling === nothing ? V : approxvoc(QgramsLookup, V, DiceDistance(); spelling...)
    end

    model = vectormodel(gw, lw, corpus, labels, V; smooth, comb)
    model = filter_tokens(model) do t
        minweight <= t.weight
    end

	X, y, dim = vectorize_corpus(model, corpus), labels, vocsize(model)
    weights = get(svm, :weights, :balanced)
    weights === :balanced && (weights = balanced_weights(y))
    kwargs = get(svm, :kwargs, NamedTuple())
    
    P = fit(projection, X, dim)
    kernel = get(svm, :kernel, :linear)
    verbose = get(svm, :verbose, verbose)
    nt = get(kwargs, :nt, Threads.nthreads())
    X_ = predict(P, X)
    
    cls = if kernel === :linear
        if size(X_, 2) > size(X_, 1)
            solver_type = LIBLINEAR.L2R_L2LOSS_SVC_DUAL
            kwargs = (; kwargs..., solver_type, weights, verbose)
            linear_train(y, X_; kwargs...)
        else
            kwargs = (; kwargs..., nt, weights, verbose, kernel=Kernel.Linear)
            svmtrain(X_, y;  kwargs...)
        end
    else 
        kwargs = (; kwargs..., nt, weights, verbose, kernel)
        svmtrain(X_, y;  kwargs...)
    end

    BagOfWordsClassifier(model, P, cls)
end

SVMPREDICT(model::LinearModel, X; nt) = linear_predict(model, X)
SVMPREDICT(model, X; nt) = svmpredict(model, X; nt)

function fit(::Type{BagOfWordsClassifier}, corpus, labels, config::NamedTuple)
    tt = config.mapfile === nothing ? IdentityTokenTransformation() : Synonyms(config.mapfile)
    fit(BagOfWordsClassifier, config.projection, corpus, labels, tt; config.collocations, config.mindocs, config.maxndocs, config.qlist, config.gw, config.lw, config.spelling, config.smooth, config.comb, config.svm)
end

function balanced_weights(y)
    C = countmap(y)
    s = sum(values(C))
    nc = length(C)
    Dict(label => (s / (nc * count)) for (label, count) in C)
end

function predict(B::BagOfWordsClassifier, corpus; nt=Threads.nthreads())
    Xtest = vectorize_corpus(B.model, corpus)
    X = predict(B.proj, Xtest)
    pred, val = SVMPREDICT(B.cls, X; nt)
    (; pred, val)
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
