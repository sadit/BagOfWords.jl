export modelselection

function modelselection(scorefun::Function, dataset, samplesize=16;
        climbing=samplesize ÷ 2,
        folds=3,
        mapfile_options=[nothing],
        gw_options = [IdfWeighting(), EntropyWeighting()],
        lw_options = [BinaryLocalWeighting(), TfWeighting()],
        collocations_options = [0, 3, 5, 7, 9, 11],
        mindocs_options = [1, 3, 5, 10, 20, 30],
        qlist_options = [[3], [2], [2, 4], [2, 5], [3, 5]],
        maxndocs_options = [0.5],
    )
    n = size(dataset, 1)
    seen = Set()
    configlist = []
    scores = []
    P = collect(1:n); shuffle!(P)
    
    randomconf() = let
        gw = rand(gw_options)
        lw = rand(lw_options)
        collocations = rand(collocations_options)
        mindocs = rand(mindocs_options)
        maxndocs = rand(maxndocs_options)
        qlist = rand(qlist_options)
        mapfile = rand(mapfile_options)
        (; gw, lw, collocations, mindocs, maxndocs, qlist, mapfile, spelling=nothing)
    end

    combine(a, b) = let
        gw = rand((a.gw, b.gw))
        lw = rand((a.lw, b.lw))
        collocations = rand((a.collocations, b.collocations))
        mindocs = rand((a.mindocs, b.mindocs)) 
        maxndocs = rand((a.maxndocs, b.maxndocs))
        qlist = rand((a.qlist, b.qlist))
        mapfile = rand((a.mapfile, b.mapfile))

        (; gw, lw, collocations, mindocs, maxndocs, qlist, mapfile, spelling=nothing)
    end

    mutate(c) = combine(c, randomconf())

    evalconfig(config, train, test) = let
        C = fit(BagOfWordsClassifier, train.text, train.klass, config)
        y = predict(C, test.text)
        scorefun(test.klass, y.pred)
    end

    for _ in 1:samplesize
        config = randomconf()
        config in seen && continue
        push!(seen, config)
        empty!(scores)
        @info "random-search> $config -- adv $(length(configlist))"
        for (itrain, itest) in kfolds(P; k=folds)
            push!(scores, evalconfig(config, dataset[itrain, :], dataset[itest, :]))
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
        for (itrain, itest) in kfolds(P; k=folds)
            push!(scores, evalconfig(config, dataset[itrain, :], dataset[itest, :]))
        end

        score = mean(scores)
        push!(configlist, (; config, score))
        sort!(configlist, by=c->c.score, rev=true)
        @info "\tperformance $score"
    end

    configlist
end
