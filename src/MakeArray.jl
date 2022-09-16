using DataFrames
using CSV
using Images
using Base: Iterators.flatten



"""
Returns X,Y position of the region.
===
Get the X,Y position (MICRON) of the region related to the whole slide image from the filename.

"""
function get_basepos(pathname)
    center_pos = split(pathname[findlast('[', pathname)+1:findlast(']', pathname)-1], ',') # center
    x = parse(Int, center_pos[1])
    y = parse(Int, center_pos[2])

    return x, y
end



"""
Creates a p_size x p_size label matrix
===
- ## Background is Zeroes.
- ## Cells are ones.

## Repositions to center(0,0) for all X,Y values in the DataFrame.
"""
function make_label_image(df_xy::DataFrame, mode::Int, o_x::Int, o_y::Int, p_size::Int, l_size::Int)

    f_size = p_size - l_size

    if mode == 0
        df_patch = filter(a -> (a.X >= (o_x + 1) && a.X <= (o_x + p_size) &&
                                a.Y >= (o_y + 1) && a.Y <= (o_y + p_size)), df_xy)
        # initialize background(zeores).
        bg = zeros(Bool, p_size, p_size)
    elseif mode == 1
        df_patch = filter(a -> (a.X >= (o_x + 1) && a.X <= (o_x + p_size) &&
                                a.Y >= (o_y + 1) && a.Y <= (o_y + f_size)), df_xy)
        # initialize background(zeores).
        bg = zeros(Bool, f_size, p_size)
    end

    # puts the cells(ones).
    for i = 1:nrow(df_patch)
        x = floor(Int, df_patch[i, :]["X"] - o_x)
        y = floor(Int, df_patch[i, :]["Y"] - o_y)

        """
        Watch out for Julia's column major here!!!
        Also, 1 pixel dot can be disappeard when shrinked.
        """
        bg[y:y+1, x:x+1] .= true
    end

    return bg
end


"""
Returns new DataFrame with X, Y positions in PIXEL
===
image res: 1860 x 1396

MICRON TO PIXEL CONVERSION:
---
``div``4 then ``times``2 to the ```[origin X, origin Y]``` to make it ```(0,0)```
---
"""
function df_xy_pixels(df_orig::DataFrame, base_x::Int, base_y::Int)::DataFrame
    new_df = df_orig[!, [:"Cell ID", :"Cell X Position", :"Cell Y Position"]]
    new_df = combine(new_df, :, [:"Cell X Position"] => (x -> abs.((base_x - 465.5) .- x) * 2) => :X)
    new_df = combine(new_df, :, [:"Cell Y Position"] => (x -> abs.((base_y - 349) .- x) * 2) => :Y)

    return new_df
end



"""
How to make a label array?
===
0. Frame mode:
```text
-------------
|_____1_____|
|  |     |  |
| 2|     |3 |
|  |     |  |
|`````4`````|
------------|
```
## concat 1 2 3 4

# Example
- feature image: 300x300
- label_width: 100
- TOTAL: 500x500
## Result with 1/2 Downscale
- top & bottom: 250x50 x2 = 25000 
- left & right: 50x150 x2 = 15000
## -> 1D vector length of 40000


1. Single side mode:
```text
______________
|      |      |
|   f  |  1   |
|______|______|
```
## concat 1

# Example
- feature image: 300x300
- label_width: 300 (->same)
- TOTAL: 300x600
## Result with 1/2 Downscale
- 150 x 150 = 22500 
## -> 1D vector length of 22500

"""
function make_label_vector(arr, mode::Int, p_size::Int, l_size::Int, label_scale::Float32)
    cell_count_before = 0
    cell_count_after = 0


    """
    ## interpolation with existing algorithms?

    https://juliamath.github.io/Interpolations.jl/stable/
    https://en.wikipedia.org/wiki/Image_scaling

    Constant(Nearest Neighbor) is not applicable... (It uses Round)
    Others are just aliasing and averages and what not...

    ## Custom Constant interpolation with CEIL?
    Any values larger than 0 is a cell after anti aliasing.

    ## Then flattens and converts to Boolean.
    """
    # convert to long vector.
    fb(x) = collect(flatten(ceil.(channelview(imresize(x, ratio = label_scale)))))

    if mode == 0
        # Makes arrays from four-sides
        top = arr[1:l_size, 1:p_size] # eg) 100x500
        left = arr[l_size+1:end-l_size, 1:l_size] # eg) 300x100
        right = arr[l_size+1:end-l_size, end-l_size+1:end] # eg) 300x100
        bot = arr[end-l_size+1:end, 1:p_size] # eg) 100x500

        for m in [top, left, right, bot]
            cell_count_before += count(i -> (i == true), m)
        end

        result = vcat(fb(top), fb(left), fb(right), fb(bot))

    elseif mode == 1
        f_size = p_size - l_size
        # Makes arrays from a side.
        side = arr[1:end, f_size+1:end]
        #println("side size: $(size(side))") # debug!

        cell_count_before += count(i -> (i == true), side)

        result = fb(side)
    end


    cell_count_after += count(i -> (i == true), result)

    # if cell_count_before != cell_count_after
    #     println("Cell counts don't match! $cell_count_before vs $cell_count_after")
    # end

    return result
end



"""
## Convert a feature image to training ready array
## Permute dims from CHW to HWC.
"""
function make_image_array(p)
    p = convert(Array{Float32}, channelview(p))
    p = permutedims(p, (2, 3, 1))
    return reshape(p, (size(p)..., 1))
end



"""
## Apply augmentation to both feature and it's label.
mode 0:
- Rotate: 90, 180, 270,
- Rotate and FlipY: 0-fy, 90-fy, 180-fy, 270-fy
-> X7 increase of data.


mode 1:
- Rotate: 180
- FlipX, FlipY
-> X3

"""
function augment_7x_process(feature, label, mode::Int, p_size::Int, l_size::Int, label_scale::Float32)
    f_set = []
    l_set = []

    if mode == 0
        # to feature
        push!(f_set, feature |> make_image_array)
        push!(f_set, rotr90(feature) |> make_image_array)
        push!(f_set, rot180(feature) |> make_image_array)
        push!(f_set, rotl90(feature) |> make_image_array)
        push!(f_set, reverse(feature, dims = 1) |> make_image_array)
        push!(f_set, reverse(rotr90(feature), dims = 1) |> make_image_array)
        push!(f_set, reverse(rot180(feature), dims = 1) |> make_image_array)
        push!(f_set, reverse(rotl90(feature), dims = 1) |> make_image_array)

        # Same augmentation for corresponding labels
        push!(l_set, make_label_vector(label, mode, p_size, l_size, label_scale))
        push!(l_set, make_label_vector(rotr90(label), mode, p_size, l_size, label_scale))
        push!(l_set, make_label_vector(rot180(label), mode, p_size, l_size, label_scale))
        push!(l_set, make_label_vector(rotl90(label), mode, p_size, l_size, label_scale))
        push!(l_set, make_label_vector(reverse(label, dims = 1), mode, p_size, l_size, label_scale))
        push!(l_set, make_label_vector(reverse(rotr90(label), dims = 1), mode, p_size, l_size, label_scale))
        push!(l_set, make_label_vector(reverse(rot180(label), dims = 1), mode, p_size, l_size, label_scale))
        push!(l_set, make_label_vector(reverse(rotl90(label), dims = 1), mode, p_size, l_size, label_scale))
    elseif mode == 1
        # mode 1
        push!(f_set, feature |> make_image_array)
        push!(f_set, rot180(feature) |> make_image_array)
        push!(f_set, reverse(feature, dims = 1) |> make_image_array)
        push!(f_set, reverse(feature, dims = 2) |> make_image_array)

        # Same augmentation for corresponding labels
        push!(l_set, make_label_vector(label, mode, p_size, l_size, label_scale))
        push!(l_set, make_label_vector(rot180(label), mode, p_size, l_size, label_scale))
        push!(l_set, make_label_vector(reverse(label, dims = 1), mode, p_size, l_size, label_scale))
        push!(l_set, make_label_vector(reverse(label, dims = 2), mode, p_size, l_size, label_scale))
    end


    return cat(f_set..., dims = 4), hcat(l_set...)
end




function augment_mode_1(feature, label, mode::Int, p_size::Int, l_size::Int, label_scale::Float32)
    f_size = p_size - l_size

    f_set = []
    l_set = []
    li_set = []

    # augment first
    orig = feature |> make_image_array
    r180 = rot180(feature) |> make_image_array
    rev1 = reverse(feature, dims = 1) |> make_image_array
    rev2 = reverse(feature, dims = 2) |> make_image_array

    # feature image
    push!(f_set, orig[1:f_size, 1:f_size, :, :])
    push!(f_set, r180[1:f_size, 1:f_size, :, :])
    push!(f_set, rev1[1:f_size, 1:f_size, :, :])
    push!(f_set, rev2[1:f_size, 1:f_size, :, :])

    # label image
    push!(li_set, orig[1:f_size, f_size+1:p_size, :, :])
    push!(li_set, r180[1:f_size, f_size+1:p_size, :, :])
    push!(li_set, rev1[1:f_size, f_size+1:p_size, :, :])
    push!(li_set, rev2[1:f_size, f_size+1:p_size, :, :])


    # Same augmentation for corresponding labels
    push!(l_set, make_label_vector(label, mode, p_size, l_size, label_scale))
    push!(l_set, make_label_vector(rot180(label), mode, p_size, l_size, label_scale))
    push!(l_set, make_label_vector(reverse(label, dims = 1), mode, p_size, l_size, label_scale))
    push!(l_set, make_label_vector(reverse(label, dims = 2), mode, p_size, l_size, label_scale))

    return cat(f_set..., dims = 4), hcat(l_set...), cat(li_set..., dims = 4)
end





# This is broken, why? (ArgumentError: reducing over an empty collection is not allowed)
function augment_7x_process_old(feature, label, mode::Int, p_size::Int, l_size::Int, label_scale::Float32)
    f(x) = x -> make_image_array(x)
    l(x) = x -> make_label_vector(x, mode, p_size, l_size, label_scale)

    # to feature
    f_set = [
        feature |> f,
        rotr90(feature) |> f,
        rot180(feature) |> f,
        rotl90(feature) |> f,
        reverse(feature, dims = 1) |> f,
        reverse(rotr90(feature), dims = 1) |> f,
        reverse(rot180(feature), dims = 1) |> f,
        reverse(rotl90(feature), dims = 1) |> f
    ]

    # Same augmentations for corresponding labels
    l_set = [
        label |> l,
        rotr90(label) |> l,
        rot180(label) |> l,
        rotl90(label) |> l,
        reverse(label, dims = 1) |> l,
        reverse(rotr90(label), dims = 1) |> l,
        reverse(rot180(label), dims = 1) |> l,
        reverse(rotl90(label), dims = 1) |> l
    ]

    return cat(f_set..., dims = 4), hcat(l_set...)
end




"""
Creates training ready arrays from single region.

"""
function region_to_arrays(img, df_xy::DataFrame; mode::Int = 0, p_size::Int = 500, l_size::Int = 100, stride::Int = 100, augment::Bool = true, label_scale::Float32 = 1 / 2)
    # declare arrays for outputs
    labels::Array = []
    features::Array = []
    label_imgs::Array = []

    num_cell_error = 0

    # feature image size
    f_size = p_size - l_size

    # image res: 1860 x 1396
    # calculate the number of x and y
    max_x = size(img)[2] - p_size
    max_y = size(img)[1] - p_size

    for o_y = 0:stride:max_y
        for o_x = 0:stride:max_x
            """
            ## select a feature image region.
            - size of feature image = patch_image - 2*label_width

            ## also select the corresponding label region from the DataFrame.
            ## Then make a binary image from it.

            patch -> 1:500, 101:600, 201:700
            feature_image -> 101:400, 201:500, 301:600
            """
            if mode == 0
                feature_img = img[o_y+l_size+1:o_y+f_size, o_x+l_size+1:o_x+f_size]

                """
                Selecting cells from the DataFrame
                """
                label_img = try
                    make_label_image(df_xy, mode, o_x, o_y, p_size, l_size)
                catch
                    @warn "Cell Data has Error! skipping..." # debug!
                    continue
                end

                if augment
                    set_feature, set_label = augment_7x_process(feature_img, label_img, mode, p_size, l_size, label_scale)
                    push!(features, set_feature)
                    push!(labels, set_label)
                else
                    push!(features, make_image_array(feature_img))
                    push!(labels, make_label_vector(label_img, mode, p_size, l_size, label_scale))
                end
            elseif mode == 1
                """
                ```text
                ______________
                |      |      |
                |  patch img  |
                |______|______|
                ```
                for each patch img & patch-label img
                    inside the augment function
                        make_label_vector
                        make_image_array
                        make_label_image
                """
                patch_img = img[o_y+1:o_y+f_size, o_x+1:o_x+p_size]
                patch_label_img = try
                    make_label_image(df_xy, mode, o_x, o_y, p_size, l_size)
                catch
                    #@warn "Cell Data has Error! skipping..." # debug!
                    # Cell Data has an error and will be skipped.
                    num_cell_error += 1
                    continue
                end

                if augment
                    set_feature, set_label, set_label_img = augment_mode_1(patch_img, patch_label_img, mode, p_size, l_size, label_scale)
                    push!(features, set_feature)
                    push!(labels, set_label)
                    push!(label_imgs, set_label_img)
                else
                    push!(features, make_image_array(patch_img[1:f_size, 1:f_size]))
                    push!(labels, make_label_vector(patch_label_img, mode, p_size, l_size, label_scale))
                    push!(label_imgs, make_image_array(patch_img[1:f_size, f_size+1:p_size]))
                end

            end
        end
    end

    # it would be empty if there was an error.
    if length(features) != 0 && length(labels) != 0
        # concat them and return
        return cat(features..., dims = 4), cat(labels..., dims = 2), cat(label_imgs..., dims = 4), num_cell_error
    else
        return false, false, false, num_cell_error
    end
end



