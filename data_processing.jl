

"""
Process & Check Data

Yongtae Kim
===

# Julia is CHW (Channels, Height, Width)


updates:
- 2021-11-24
1. uses built in augmentation (Augmentor changes matrix size... wtf?)
2. boolean early
successful run

- 2021-11-26
fixing label-feature mismatch...
recover_label_img has problem... (or even make_label_vector!!)

So far...
1. make_label_image -> augment_7x -> make_label_vector -> recover_label_img
TODO:
findall
coodinate problem?
col major ?

draw coodinate

[0,0]
-----------------> [ ,x]
|
|
|
[y, ]

df to x,y draw has problem? (make_label_image)
update: YES. make_label_image was the problem. (x,y -> y,x)

- 2021-11-29
1. Cleaned up codes.
2. Main loop for hdf5.
"""




include("./src/CheckArray.jl")
include("./src/MakeArray.jl")


using DataFrames
using CSV
using Images
using ProgressMeter
using HDF5
using Parameters: @with_kw



@with_kw struct Args

    """
    0. Frame mode:
    ```text
    -------------
    |_____1_____|
    |  |     |  |
    | 2|  f  |3 |
    |  |     |  |
    |`````4`````|
    ------------|
    ```

    1. Single side mode:
    ```text
    ______________
    |      |      |
    |   f  |  1   |
    |______|______|
    ```
    """

    mode::Int = 1 # 0 for frame mode, 1 for single side mode.

    path_gpu::String = "/home/yongtae/data/datasets/LCNEC/batch_v2_level2/"
    path_ncc::String = "/home/yongtae/Programming/building/2021-11-25 - TN1/"
    path_home::String = "/home/yongtae/Programming/building/2021-11-25 - TN1/"
    path = path_gpu

    size_label::Int = 288
    size_feature::Int = 288

    stride::Int = 288
    # 0~1 cell lost during the 1/2 downscale process.
    scale_ratio::Float32 = 1 / 3
    augment::Bool = true

    hdf5 = "lcnec_mode-$(mode)_aug-$(augment)_f$(size_feature)-l$(size_label)-st$(stride)_level1-2_1.h5" #"lcnec_aug-x7_p500-l100-st250_level1-2_comp4.h5"
end



function main()
    a = Args()
    # debugging
    if isfile(a.path * a.hdf5)
        rm(a.path * a.hdf5)
    end
    h5f = h5open(a.path * a.hdf5, "cw")
    g = create_group(h5f, "all")

    tot_cell_errors = 0

    if a.mode == 0
        size_patch = a.size_feature + a.size_label * 2
        """
        - top & bottom: 250x50 x2 = 25000
        - left & right: 50x150 x2 = 15000
        """
        v_size = floor(Int, (size_patch * a.scale_ratio) * (a.size_label * a.scale_ratio) * 2 + (a.size_label * a.scale_ratio) * (a.size_feature * a.scale_ratio) * 2)

    elseif a.mode == 1
        size_patch = a.size_feature + a.size_label
        """
        Width x Height
        """
        v_size = floor(Int, (a.size_feature * a.scale_ratio) * (a.size_label * a.scale_ratio))
    end

    # Creating datasets on the HDF5
    ds_f = create_dataset(g, "features", datatype(Float32),
        ((a.size_feature, a.size_feature, 3, 0), (a.size_feature, a.size_feature, 3, -1)), chunk = (a.size_feature, a.size_feature, 3, 8), compress = 4)

    ds_l = create_dataset(g, "labels", datatype(Float32),
        ((v_size, 0), (v_size, -1)), chunk = (v_size, 8), compress = 4)

    ds_li = create_dataset(g, "label_imgs", datatype(Float32),
        ((a.size_feature, a.size_feature, 3, 0), (a.size_feature, a.size_feature, 3, -1)), chunk = (a.size_feature, a.size_feature, 3, 8), compress = 4)


    scans = [d for d in readdir(a.path, join = true) if isdir(d)]
    # p_scan = Progress(length(scans), 1, "Processing Slides...")

    # a batch level dir
    for i = 1:length(scans)
        scan = scans[i]

        println("Current Slide ( $i / $(length(scans)) ): $scan")

        # inform outputs
        # raw image for now (not the composite)
        regions = [f for f in readdir(scan, join = true) if endswith(f, "].tif")]
        p_region = Progress(length(regions), 1, join(["Processing Regions..."]))
        next!(p_region)

        for r in regions
            # current size of dataset.
            s_tot = size(ds_f)[4]
            #next!(p_region; showvalues = [(:s_tot, s_tot)])

            img = load(r)
            df = df_xy_pixels(DataFrame(CSV.File(
                    replace(r, "].tif" => "]_cell_seg_data.txt"), delim = '\t')),
                get_basepos(r)[1], get_basepos(r)[2])

            # process a region.
            features, labels, label_imgs, num_cell_errors = region_to_arrays(img, df, mode = a.mode, p_size = size_patch, l_size = a.size_label, stride = a.stride, augment = a.augment, label_scale = a.scale_ratio)

            tot_cell_errors += num_cell_errors

            next!(p_region; showvalues = [(:s_tot, s_tot), (:tot_cell_errors, tot_cell_errors)])

            """
            # Writing to HDF5 File
            Chunking size by augmentation??:
            feature: 300,300,3,8
            label: 40000x8 = 320000?
            Or region?
            f: 300,300,3, ?
            label: 40000x?
            """
            # region_to_arrays will return false if there was an error.
            if features != false && labels != false
                # size to be added
                s_in = size(features)[4]

                # Extends them first
                # length must be same for feature and label.
                HDF5.set_extent_dims(ds_f, (a.size_feature, a.size_feature, 3, s_tot + s_in))
                HDF5.set_extent_dims(ds_l, (v_size, s_tot + s_in))
                HDF5.set_extent_dims(ds_li, (a.size_feature, a.size_feature, 3, s_tot + s_in))

                # Append
                # println("s_in: $(s_in), labels: $(size(labels)), features: $(size(features))") # debug!
                ds_f[:, :, :, s_tot+1:s_tot+s_in] = features
                ds_l[:, s_tot+1:s_tot+s_in] = labels
                ds_li[:, :, :, s_tot+1:s_tot+s_in] = label_imgs
            else
                println("An entire region had an error and was skipped...")
            end
        end
        # next!(p_scan; showvalues = [(:d, d)])
        println()
    end

    close(h5f)
end


main()