"""
LOSS, ACCURACY...
"""

begin
    using Flux.Losses: logitcrossentropy, binarycrossentropy
    using Flux: onecold
    using Flux
end



function eval_loss_accuracy(loader, model, device)
    l = 0.0f0
    acc = 0
    ntot = 0
    for idx in loader
        x, y = load_xy(idx)
        x = x |> device
        y = y |> device
        ŷ = model(x)
        l += loss(ŷ, y) * size(x)[end]
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return (loss = l / ntot |> round4, acc = acc / ntot * 100 |> round4)
end

loss(ŷ, y) = logitcrossentropy(ŷ, y)

round4(x) = round(x, digits = 4)

l = 0.0f0
acc = 0
ntot = 0

function new_eval(loader, model, device)
    l = 0.0f0
    acc = 0
    ntot = 0
    for idx in loader
        x, y = load_xy(idx)
        x = x |> device
        y = y |> device
        ŷ = model(x)
        l += loss(ŷ, y) * length(y)
        acc += sum((ŷ |> cpu) .== (y |> cpu))
        ntot += length(y)
    end
    return (loss = l / ntot |> round4, acc = acc / ntot * 100 |> round4)
end


"""
New calculations:
Ignore the values where it correctly predicted as backround(zero).

"""
function new_eval2(loader, model, device)
    l = 0.0f0
    # n_total = 0
    cells_total = 0
    cells_correct = 0

    for idx in loader
        x, y = load_xy(idx)
        #x = x |> device
        #y = y |> device
        ŷ = model(x)

        # LOSS
        l += loss(ŷ, y)

        # ACC
        sum_cells = (ŷ + y)
        cells_total += count(x -> (x == 2) || (x == 1), sum_cells)
        cells_correct += count(x -> (x == 2), sum_cells)

        # n_total += length(y)
    end

    return (loss = l |> round4, acc = (cells_correct / cells_total) * 100 |> round4)
end



"""
New calculations:
Ignore the values where it correctly predicted as backround(zero).

Evaluation: Loss, Accuracy
Eg.
pred:
0.2, 0.9, 0.1, 0.9
label:
0.0, 1.0, 1.0, 0.0
sum
0.2, 1.9, 1.1, 0.9



"""
function new_eval3(loader, model, device)
    l = 0.0f0
    # n_total = 0
    cells_total = 0
    cells_correct = 0
    #cells_probabilities = 0

    for idx in loader
        x, y = load_xy(idx)
        x = x |> device
        y = y |> device
        ŷ = model(x)

        # LOSS
        l += loss(ŷ, y)

        # ACC
        sum_cells = (ŷ |> cpu) + (y |> cpu)
        cells_total += count(x -> (x > 1.9 && x <= 2.0) || (x > 0.9 && x <= 1.0), sum_cells)
        cells_correct += count(x -> (x > 1.9 && x <= 2.0), sum_cells)

        # Optional
        #cells_probabilities += sum(ŷ |> cpu)
        # n_total += length(y)
    end

    return (loss = l |> round4, acc = (cells_correct / cells_total) * 100 |> round4)
end




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
function new_eval4(loader, model, device)
    l = 0.0f0
    # n_total = 0
    cells_pred = 0
    cells_label = 0
    cells_correct = 0
    #cells_probabilities = 0

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

        # change values inside the range to 1.0
        sum_cells = (y + replace(x -> (x > 0.9 && x <= 1.0) ? 1.0 : x, ŷ))
        # count them
        cells_pred += count(x -> (x == 1.0), ŷ)
        cells_label += count(x -> (x == 1.0), y)
        # cells_total += count(x -> (x == 1.0) || (x == 2.0), sum_cells)
        # 2.0s are the correct predictions
        cells_correct += count(x -> (x == 2.0), sum_cells)

        # Optional
        #cells_probabilities += sum(ŷ |> cpu)
        # n_total += length(y)
    end

    return (loss = l |> round4,
        pred = cells_pred,
        label = cells_label,
        correct = cells_correct)
end



"""
batch.
"""
function new_eval5(loader, model, device)
    l = 0.0f0
    # n_total = 0
    cells_pred = 0
    cells_label = 0
    cells_correct = 0
    #cells_probabilities = 0


    x, y = load_xy(idx)
    x = x |> device
    y = y |> device
    ŷ = model(x)

    # LOSS
    l += loss(ŷ, y)

    # ACC
    y = y |> cpu
    ŷ = ŷ |> cpu

    for i in range(1, stop = length(y))
        # change values inside the range to 1.0
        sum_cells = (y[i] + replace(x -> (x > 0.9 && x <= 1.0) ? 1.0 : x, ŷ[i]))
        # count them
        cells_pred += count(x -> (x == 1.0), ŷ[i])
        cells_label += count(x -> (x == 1.0), y[i])
        # cells_total += count(x -> (x == 1.0) || (x == 2.0), sum_cells)
        # 2.0s are the correct predictions
        cells_correct += count(x -> (x == 2.0), sum_cells)
    end


    return (loss = l |> round4,
        pred = cells_pred,
        label = cells_label,
        correct = cells_correct,
        acc = (cells_correct / (cells_label + cells_pred)) |> round4)
end





y = rand(Bool, 40000)
ŷ = rand(Float32, 40000)
"""
LOSS

"""

begin
    y = rand(Bool, 20)
    ŷ = rand(Bool, 20)
end
f = rand(Float32, 20)

onecold(f)

softmax(x, dims) = exp.(x) ./ sum(exp.(x), dims = dims)

logitcrossentropy(ŷ, y)
logitcrossentropy(ŷs, ys)

logitcrossentropy(y, y)
logitcrossentropy(ŷs, ŷs)

st = softmax(ŷ, 1)

Flux.crossentropy(st, y)


"""
ACC

"""

y .== 0
sum(onecold(y) .== onecold(ŷ))

sum(ŷ .== y) / 7 * 100 |> round4

(ŷ .== y)
y
ŷ

ŷ .== 0
(y .== 0)

(y .== 0) .== (ŷ .== 0)


"""
(ŷ + y)
0: background. ignore
1: wrong prediction of cell
2: correct prediction
"""

vacc = (ŷ + y)


count(x -> (x == 2) || (x == 1), vacc)