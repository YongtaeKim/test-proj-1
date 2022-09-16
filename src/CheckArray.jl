include("MakeArray.jl")

using Plots
using Images
using ImageDraw
using Colors
using ImageShow






"""
Recreates binary label image from label vector
===

"""
function recover_label_img(label, mode::Int, p_size::Int, l_size::Int, ratio::Float64)
    # calculate each lengths from the sizes
    # - top & bottom: 50x250 x2 = 25000 
    # - left & right: 150x50 x2 = 15000
    ps = floor(Int, p_size * ratio) # 250
    ls = floor(Int, l_size * ratio) # 50

    if mode == 0
        fs = ps - 2 * ls # 150
        len_tb = ps * ls
        len_side = ls * fs

        # Cut out the vectors.
        top = label[1:len_tb]
        left = label[len_tb+1:len_tb+len_side]
        right = label[len_tb+len_side+1:len_tb+len_side+len_side]
        bot = label[len_tb+len_side+len_side+1:len_tb+len_side+len_side+len_tb]

        # Un-Flatten the vectors.
        top = reshape(top, (ls, ps)) # 50x250
        left = reshape(left, (fs, ls)) # 150x50
        right = reshape(right, (fs, ls))
        bot = reshape(bot, (ls, ps))

        # recover the boolean image
        label_img = zeros(Bool, (ps, ps))
        label_img[1:ls, 1:ps] = top
        label_img[ls+1:ls+fs, 1:ls] = left
        label_img[ls+1:ls+fs, ls+fs+1:end] = right
        label_img[ls+fs+1:end, 1:ps] = bot
    elseif mode == 1
        # just a matrix
        fs = ps - ls # 150
        # Un-Flatten the vectors.
        label_img = reshape(label, (fs, ls)) # 50x250
    end

    return label_img
end



"""
Recover patch image
===
Recover patch image(feature image + label region) for checking purpose.
(But it can't recover the region from dataset??)
"""
function recover_patch_img(img, num; p_size::Int = 500, l_size::Int = 100, stride::Int = 99, augment::Bool = true)
    count = 1
    max_x = size(img)[2] - p_size
    max_y = size(img)[1] - p_size

    for o_y = 0:stride:max_y
        for o_x = 0:stride:max_x

            # Found a match
            if (count <= num) && (count + 8 > num) # count <= num < count+8 
                patch_img = copy(img[o_y+1:o_y+p_size, o_x+1:o_x+p_size])
                # determines the augmentation
                aug_mode = num - count
                println("Count:$count, num:$num, Mode:$aug_mode")
                if augment
                    if aug_mode == 0
                        return patch_img
                        # do augmentation
                    elseif aug_mode == 1
                        return rotr90(patch_img)
                    elseif aug_mode == 2
                        return rot180(patch_img)
                    elseif aug_mode == 3
                        return rotl90(patch_img)
                    elseif aug_mode == 4
                        return reverse(patch_img, dims = 1)
                    elseif aug_mode == 5
                        return reverse(rotr90(patch_img), dims = 1)
                    elseif aug_mode == 6
                        return reverse(rot180(patch_img), dims = 1)
                    elseif aug_mode == 7
                        return reverse(rotl90(patch_img), dims = 1)
                    end
                    #else
                    #    return patch_img
                end

                # no match
            elseif augment
                # count addtional 7 augmented patches per patch
                count += 8
            else # no match
                count += 1
            end
        end
    end

    println("Unable to find!")
    return
end



"""
Check out a pair of label, feature image set.
===
Recovers a pair of label, feature image set from the dataset then plots on the patch image.
Useful for checking out dataset just in case.
"""
function check_label_patch(img, l_arr, f_arr, num;
    p_size::Int = 500, l_size::Int = 100, stride::Int = 99, augment::Bool = true, ratio = 1 / 2)
    r = 1 / ratio
    # acquire each image
    l_img = recover_label_img(l_arr[:, num])
    p_img = recover_patch_img(img, num)
    f_img = colorview(RGB, permutedims(f_arr[:, :, :, num], (3, 1, 2)))

    # Draw some outline on the input image
    f_img[:, 1] .= RGB{N0f8}(1, 0, 0)
    f_img[:, end] .= RGB{N0f8}(1, 0, 0)
    f_img[1, :] .= RGB{N0f8}(1, 0, 0)
    f_img[end, :] .= RGB{N0f8}(1, 0, 0)

    """
    Draw dots for cells
    TODO: Is simply doubling the XY correct?
    Maybe substract 1 too?
    """
    for i in findall(l_img)
        x = floor(Int, i[2] * r)# - 1
        y = floor(Int, i[1] * r) - 1

        draw!(p_img, Ellipse(CirclePointRadius(x, y, 2)), RGB{N0f8}(0, 0, 1))
        #p_img[y,x] = RGB{N0f8}(0, 0, 1)
    end

    p_img[l_size+1:p_size-l_size, l_size+1:p_size-l_size] = f_img

    # p_img[:,l_size+1] .= RGB{N0f8}(0, 1, 0)
    # p_img[:,p_size-l_size] .= RGB{N0f8}(0, 1, 0)
    # p_img[l_size+1,:] .= RGB{N0f8}(0, 1, 0)
    # p_img[p_size-l_size,:] .= RGB{N0f8}(0, 1, 0)

    return p_img
end





