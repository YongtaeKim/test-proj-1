"""
2021-11-09
data prepared.
Lenet5. 39% max.
2021-11-11
Fixed the HDF5 failure. HDF5+Dataloader must not be in a function. (must be global?)
2021-12-02
fixed out of memory with load_xy.
2021-12-03
Lowered the dataset & batch size.
shuffle early.
Implemented saving pretrained weight in hdf5.
Now using git.

"""

include("src/models.jl")
include("src/utils.jl")


using HDF5
using Flux
using Flux.Data: DataLoader

using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold, early_stopping

using Statistics, Random
using Logging: with_logger
using ProgressMeter: @showprogress
using Parameters: @with_kw

using BSON
using Random

using CUDA








@with_kw struct Args
    η::Float64 = 1e-4 #3e-4            # learning rate
    size::Int = 3230 # size of dataset. Current max: 726048 - 26048.
    λ = 0                # L2 regularizer param, implemented as weight decay
    batchsize::Int = 12 #16      # batch size
    epochs::Int = 64          # number of epochs
    seed = 0             # set seed > 0 for reproducibility
    use_cuda::Bool = true      # if true use cuda (if available)
    infotime::Int = 1      # report every `infotime` epochs
    checktime::Int = 2#3        # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    tblogger::Bool = false      # log training with tensorboard


    # saving weights
    savepath = "/home/yongtae/data/documents/test/tn/runs/" # path for trained weights file
    savename = "weights_tn2a_mae-loss_d$(size)_b$(batchsize)_rl$(η)_1" # name for trained weights file


    # path for dataset
    path_home = ""
    path_gpu = "/home/yongtae/data/datasets/LCNEC/test/"
    path_ncc = ""
    path = path_gpu
    h5file = path * "lcnec_mode-1_aug-true_f288-l288-st288_test_7.h5"
end






"""
# Loads the hdf5 and data.
The hdf5 does not work if it's inside a function.
Must be loaded globaly.
"""
f = h5open(Args().h5file, "r")






"""
# INDEXING METHOD
Basic method fails due to out of memory

leave 26048 out of 726048 for evaluation.
"""
len = size(f["all"]["features"])[4]
idx = collect(Int, 1:3230)
shuffle!(idx)

len_test = Args().size ÷ 15

idx_test = idx[1:len_test]
idx_train = idx[len_test+1:Args().size]

test_loader = DataLoader(idx_test, batchsize = Args().batchsize, shuffle = true)
train_loader = DataLoader(idx_train, batchsize = Args().batchsize, shuffle = true)







function train(; kws...)
    args = Args()
    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.use_cuda && CUDA.functional()

    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    ## DATA
    # train_loader, test_loader = get_data(args)
    @info "Dataset: $(train_loader.nobs) train and $(test_loader.nobs) test examples"


    """
    SELECT THE MODEL HERE!
    """
    model = tn_2a() |> device
    @info "model: $(num_params(model)) trainable params"


    ps = Flux.params(model)


    opt = ADAM(args.η)
    if args.λ > 0 # add weight decay, equivalent to L2 regularization
        opt = Optimiser(opt, WeightDecay(args.λ))
    end

    ## LOGGING UTILITIES
    if args.tblogger
        tblogger = TBLogger(args.savepath, tb_overwrite)
        set_step_increment!(tblogger, 0) # 0 auto increment since we manually set_step!
        @info "TensorBoard logging at \"$(args.savepath)\""
    end

    function report(epoch)
        train = eval_loss_accuracy(train_loader, model, device)
        test = eval_loss_accuracy(test_loader, model, device)
        println("Epoch: $epoch   Test: $(test)   Train: $(train)")

        if args.tblogger
            set_step!(tblogger, epoch)
            with_logger(tblogger) do
                #@info "train" loss = train.loss acc = train.acc
                @info "test" loss = test.loss acc = test.acc
            end
        end

        return test.acc
    end

    ## TRAINING
    @info "Start Training"
    report(0)
    for epoch = 1:args.epochs
        @showprogress for idx in train_loader
            x, y = load_xy(idx)
            x = x |> device
            y = y |> device
            gs = Flux.gradient(ps) do
                ŷ = model(x)
                loss(ŷ, y)
            end

            Flux.Optimise.update!(opt, ps, gs)
        end

        ## Printing and logging
        epoch % args.infotime == 0 && report(epoch)
        if args.checktime > 0 && epoch % args.checktime == 0
            !ispath(args.savepath) && mkpath(args.savepath)

            modelpath = join([args.savepath, args.savename, "_ep_", string(epoch), ".h5"])
            #modelpath = joinpath(args.savepath, args.savename, "_ep_", string(epoch), ".h5")
            let model = cpu(model) # return model to cpu before serialization
                #BSON.@save modelpath model epoch
                save_weights(model, modelpath)
            end
            @info "Model saved in \"$(modelpath)\""
        end
    end
end

train()

close(f)