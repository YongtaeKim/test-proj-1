using HDF5
using Flux
using Flux: loadparams!
using Flux.Losses: logitcrossentropy, mae





"""
Loss Function

"""
loss(ŷ, y) = mae(ŷ, y)
#loss(ŷ, y) = logitcrossentropy(ŷ, y)





"""
Custom Sigmoid Activation Function

"""
my_activation(x) = sigmoid(20 * (x - 0.5))






"""
Evaluation: Loss, Accuracy
Eg.
- pred:
0.2, 1.0, 0.6, 1.0
- label:
0.0, 1.0, 1.0, 0.0
- sum:
0.2, 2.0, 1.6, 1.0

- ACC => 1 / 3


"""
function eval_loss_accuracy(loader, model, device)
    l = 0.0f0
    # n_total = 0
    cells_pred = 0
    cells_label = 0
    cells_correct = 0
    #cells_probabilities = 0

    #pred_size = 0
    # not working...?
    pred_zeroes = 0
    pred_nans = 0


    for idx in loader
        x, y = load_xy(idx)
        x = x |> device
        y = y |> device
        ŷ = model(x)

        # LOSS
        l += loss(ŷ, y)

        # ACC
        y = y |> cpu
        ŷ = ŷ |> cpu

        y = vcat(y...)
        ŷ = vcat(ŷ...)

        # change values that in the range to 1.0
        replace!(x -> (x > 0.5 && x <= 1.0) ? 1.0 : x, ŷ) # 0.9 to 0.5 to looking for hope.

        # count them
        cells_pred += count(x -> (x == 1.0), ŷ)
        cells_label += count(x -> (x == 1.0), y)

        sum_cells = y + ŷ
        # 2.0s are the correct predictions
        cells_correct += count(x -> (x == 2.0), sum_cells)

        #pred_size += length(ŷ)
        pred_zeroes += count(x -> (x < 0.0001 && x >= 0.0), ŷ)
        pred_nans += count(x -> isnan(x), ŷ)
    end


    return (loss = l |> round4,
        pred = cells_pred,
        label = cells_label,
        hope = cells_correct,
        acc = (cells_correct / (cells_label + cells_pred)) |> round4,
        #p_all = pred_size,
        p_zero = pred_zeroes,
        p_nan = pred_nans)
end





# utility functions
num_params(model) = sum(length, Flux.params(model))
round4(x) = round(x, digits = 4)





"""
Save the model weights into HDF5 file.
"""
function save_weights(model, file)
    weights = collect(params(cpu(model)))

    f = h5open(file, "w")
    g = create_group(f, "weights")

    for i = 1:length(weights)
        #create_dataset(g, string(i), weights[i][]) # this stores only zeroes!!
        g[string(i)] = weights[i]
    end

    close(f)

    return
end





"""
Save the model weights from HDF5 file to the model.
"""
function load_weights(model, file)
    weights = []

    f = h5open(file, "r")
    g = f["weights"]

    for i = 1:length(g)
        push!(weights, g[string(i)][])
    end

    loadparams!(model, weights)

    return model

end





"""
Loads data with list of index that is a batch.
"""
function load_xy(idxs)
    x = []
    y = []
    for i in idxs
        push!(x, reshape(f["all"]["features"][:, :, :, i], (288, 288, 3, 1)))
        push!(y, reshape(f["all"]["labels"][:, i], (9216, 1)))
    end

    x = cat(x..., dims = 4)
    x = Array{Float32,4}(x)

    y = cat(y..., dims = 2)

    # 40000 vector
    y = Matrix{Float32}(y)

    # OR 200x200 matrix
    #y = reshape(y, (200, 200, 1, length(idxs)))
    #y = convert(Array{Float32,4}, y)

    return x, y
end


function load_xy1(idxs)
    x = []
    y = []
    for i in idxs
        push!(x, reshape(f["all"]["features"][:, :, :, i], (300, 300, 3, 1)))
        # Going crazy!
        p = reshape(f["all"]["labels"][:, i], (40000, 1))
        p = 10000 .* p
        push!(y, p)
        # push!(y, reshape(f["all"]["labels"][:, i], (40000, 1)))
    end

    x = cat(x..., dims = 4)
    x = Array{Float32,4}(x)

    y = cat(y..., dims = 2)

    # 40000 vector
    #y = Matrix{Float32}(y)

    # OR 200x200 matrix
    y = reshape(y, (200, 200, 1, length(idxs)))
    y = convert(Array{Float32,4}, y)

    return x, y
end