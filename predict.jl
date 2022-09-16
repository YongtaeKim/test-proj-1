"""
1. Load the pretrained weight.
2. Predict.
3. Plot it.
4. Compare with the label.

"""


using HDF5
using Flux

include("./src/models.jl")
include("./src/CheckArray.jl")
include("./src/utils.jl")




begin
    h5_data = "/home/yongtae/data/datasets/LCNEC/test/lcnec_mode-1_aug-true_f288-l288-st288_test_7.h5"

    f = h5open(h5_data, "r")
    arr_l = f["all"]["labels"]
    arr_f = f["all"]["features"]
    arr_li = f["all"]["label_imgs"]
end

begin
    model = tn_1f() # Select the model here.
    # The pretrained weight.
    h5_weights = "/home/yongtae/data/documents/test/tn/runs/weights_tn1f_mae-loss_d20000_b24_rl0.0001_3_ep_4.h5"

    loaded_model = load_weights(model, h5_weights)
end

size(arr_l)
size(arr_f)
size(arr_li)

# Pick an example to test
test_num = 1110


colorview(RGB, permutedims(arr_li[:, :, :, test_num], (3, 1, 2)))
colorview(Gray, recover_label_img(arr_l[:, test_num], 1, 288 + 288, 288, 1 / 3))
colorview(RGB, permutedims(arr_f[:, :, :, test_num], (3, 1, 2)))




a = zeros(9, 9)
a[4:5, 4:5] .= 1.0
colorview(Gray, a)

a3 = imresize(a, ratio = 1 / 3)

colorview(Gray, a3)




begin
    test_f = reshape(arr_f[:, :, :, 23464], (300, 300, 3, 1))
    pred = loaded_model(test_f)
end

maximum(pred)

count(x -> (x > -2.2876814f-5), pred)

count(x -> (x == 1.0), arr_l[:, test_num])

pred_p = replace(x -> (x > -2.2876814f-5) ? 1.0 : 0, pred)

begin
    weights = collect(params(cpu(loaded_model)))
    #pred_img = colorview(Gray, recover_label_img(pred555))

    for a in weights
        println(count(x -> isnan(x), a))
    end
end

loaded_model(rand(Float32, 300, 300, 3, 1))

