export modelselection

function modelselection(scorefun::Function, text, labels, samplesize=16;
        climbing=samplesize ÷ 2,
        folds=3,
        validation_text=nothing,
        validation_labels=nothing,
        mapfile_options=[nothing],
        projection_options = [RawVectors()],
        gw_options = [IdfWeighting(), EntropyWeighting()],
        lw_options = [BinaryLocalWeighting()],
        collocations_options = [7],
        mindocs_options = [3],
        maxndocs_options = [1.0],
        smooth_options = [0, 0.1],
        comb_options = [SigmoidPenalizeFewSamples(), NormalizedEntropy()],
        qlist_options = [[2, 4], [2, 5], [3, 5]],
        minweight_options = [1e-4],
        svm_options = [(; kernel=:linear)]
    )

    n = length(text)
    seen = Set()
    configlist = []
    scores = []
    P = collect(1:n); shuffle!(P)
    
    randomconf() = let
        projection = rand(projection_options)
        gw = rand(gw_options)
        lw = rand(lw_options)
        collocations = rand(collocations_options)
        mindocs = rand(mindocs_options)
        maxndocs = rand(maxndocs_options)
        svm = rand(svm_options)
        if gw isa EntropyWeighting
            smooth = rand(smooth_options)
            comb = rand(comb_options)
        else
            smooth = 0
            comb = nothing
        end

        minweight = rand(minweight_options)
        qlist = rand(qlist_options)
        mapfile = rand(mapfile_options)

        (; projection, gw, lw, collocations, mindocs, maxndocs, smooth, comb, minweight, qlist, svm, mapfile, spelling=nothing)
    end

    combine(a, b) = let
        projection = rand((a.projection, b.projection))
        gw = rand((a.gw, b.gw))
        lw = rand((a.lw, b.lw))
        collocations = rand((a.collocations, b.collocations))
        mindocs = rand((a.mindocs, b.mindocs)) 
        maxndocs = rand((a.maxndocs, b.maxndocs))
        svm = rand((a.svm, b.svm))
        if gw isa EntropyWeighting
            smooth = rand(smooth_options)
            comb = rand(comb_options)
        else
            smooth = 0
            comb = nothing
        end
        minweight = rand((a.minweight, b.minweight))
        qlist = rand((a.qlist, b.qlist))
        mapfile = rand((a.mapfile, b.mapfile))

        (; projection, gw, lw, collocations, mindocs, maxndocs, smooth, comb, minweight, qlist, svm, mapfile, spelling=nothing)
    end

    mutate(c) = combine(c, randomconf())

    evalconfig(config, train_text, train_labels, test_text, test_labels) = let
        C = fit(BagOfWordsClassifier, train_text, train_labels, config)
        y = predict(C, test_text)
        scorefun(test_labels, y.pred)
    end

    FOLDS = collect(kfolds(P; k=folds))

    for _ in 1:samplesize
        config = randomconf()
        config in seen && continue
        push!(seen, config)
        empty!(scores)
        @info "random-search> $config -- adv $(length(configlist))"
        if validation_text === nothing
            for (itrain, itest) in FOLDS
                push!(scores, evalconfig(config, text[itrain], labels[itrain], text[itest], labels[itest]))
            end
        else
            push!(scores, evalconfig(config, text, labels, validation_text, validation_labels))
        end

        score = mean(scores)
        push!(configlist, (; config, score))
        @info "\tperformance $score"
    end
    sort!(configlist, by=c->c.score, rev=true)

    for _ in 1:climbing
        config = mutate(first(configlist).config)
        config in seen && continue
        push!(seen, config)
        empty!(scores)
        @info "hill-climbing> $config -- adv $(length(configlist))"
        if validation_text === nothing
            for (itrain, itest) in FOLDS
                push!(scores, evalconfig(config, text[itrain], labels[itrain], text[itest], labels[itest]))
            end
        else
            push!(scores, evalconfig(config, text, labels, validation_text, validation_labels))
        end

        score = mean(scores)
        push!(configlist, (; config, score))
        sort!(configlist, by=c->c.score, rev=true)
        @info "\tperformance $score"
    end

    configlist
end
