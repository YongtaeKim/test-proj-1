




using Flux
using CUDA



# ================================================================================ #


function tn_2a(; imgsize = (288, 288, 3))

    Chain( # 288
        Conv((21, 21), 3 => 8, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(8),
        Conv((21, 21), 8 => 8, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(8),
        Conv((21, 21), 8 => 8, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(8),
        Conv((21, 21), 8 => 8, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(8),
        Conv((21, 21), 8 => 8, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(8),
        Conv((21, 21), 8 => 8, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(8),
        Conv((21, 21), 8 => 8, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(8),
        Conv((21, 21), 8 => 8, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(8),
        # 144
        Conv((17, 17), 8 => 16, relu, pad = SamePad(), stride = (2, 2)),
        BatchNorm(16),
        Conv((17, 17), 16 => 16, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(16),
        Conv((17, 17), 16 => 16, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(16),
        Conv((17, 17), 16 => 16, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(16),
        Conv((17, 17), 16 => 16, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(16),
        Conv((17, 17), 16 => 16, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(16),
        Conv((17, 17), 16 => 16, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(16),
        Conv((17, 17), 16 => 16, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(16),
        # 72
        Conv((13, 13), 16 => 32, relu, pad = SamePad(), stride = (2, 2)),
        BatchNorm(32),
        Conv((13, 13), 32 => 32, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(32),
        Conv((13, 13), 32 => 32, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(32),
        Conv((13, 13), 32 => 32, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(32),
        Conv((13, 13), 32 => 32, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(32),
        Conv((13, 13), 32 => 32, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(32),
        Conv((13, 13), 32 => 32, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(32),
        Conv((13, 13), 32 => 32, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(32),
        # 24
        Conv((9, 9), 32 => 64, relu, pad = SamePad(), stride = (3, 3)),
        BatchNorm(64),
        Conv((9, 9), 64 => 64, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(64),
        Conv((9, 9), 64 => 64, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(64),
        Conv((9, 9), 64 => 64, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(64), # 48
        ConvTranspose((8, 8), 64 => 32, stride = (2, 2), sigmoid, pad = SamePad()),
        BatchNorm(32), # 96
        ConvTranspose((16, 16), 32 => 16, stride = (2, 2), sigmoid, pad = SamePad()),
        BatchNorm(16), # 192
        ConvTranspose((32, 32), 16 => 1, stride = (2, 2), sigmoid, pad = SamePad()),
        BatchNorm(1),
        Conv((1, 1), 1 => 1, pad = SamePad(), stride = 1),
        Conv((1, 1), 1 => 1, pad = SamePad(), stride = 1),
        Conv((1, 1), 1 => 1, pad = SamePad(), stride = 1),
        Conv((1, 1), 1 => 1, pad = SamePad(), stride = 1),
        Flux.flatten,
        Dense(36864, 9216)
    )
end





function tn_1f(; imgsize = (300, 300, 3))

    Chain( # 300, 300
        Conv((21, 21), imgsize[end] => 8, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(8),
        Conv((21, 21), 8 => 8, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(8),
        Conv((21, 21), 8 => 8, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(8),
        Conv((21, 21), 8 => 8, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(8),
        Conv((21, 21), 8 => 8, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(8),
        Conv((21, 21), 8 => 8, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(8),
        Conv((21, 21), 8 => 8, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(8),
        Conv((21, 21), 8 => 8, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(8),
        # 150, 150
        Conv((17, 17), 8 => 16, relu, pad = SamePad(), stride = (2, 2)),
        BatchNorm(16),
        Conv((17, 17), 16 => 16, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(16),
        Conv((17, 17), 16 => 16, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(16),
        Conv((17, 17), 16 => 16, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(16),
        Conv((17, 17), 16 => 16, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(16),
        Conv((17, 17), 16 => 16, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(16),
        Conv((17, 17), 16 => 16, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(16),
        Conv((17, 17), 16 => 16, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(16),
        # 75, 75
        Conv((13, 13), 16 => 32, relu, pad = SamePad(), stride = (2, 2)),
        BatchNorm(32),
        Conv((13, 13), 32 => 32, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(32),
        Conv((13, 13), 32 => 32, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(32),
        Conv((13, 13), 32 => 32, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(32),
        Conv((13, 13), 32 => 32, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(32),
        Conv((13, 13), 32 => 32, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(32),
        Conv((13, 13), 32 => 32, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(32),
        Conv((13, 13), 32 => 32, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(32),
        # 25, 25
        Conv((9, 9), 32 => 64, relu, pad = SamePad(), stride = (3, 3)),
        BatchNorm(64),
        Conv((9, 9), 64 => 64, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(64),
        Conv((9, 9), 64 => 64, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(64),
        Conv((9, 9), 64 => 64, relu, pad = SamePad(), stride = (1, 1)),
        BatchNorm(64), # 50, 50
        ConvTranspose((8, 8), 64 => 32, stride = (2, 2), sigmoid, pad = SamePad()),
        BatchNorm(32), # 100, 100
        ConvTranspose((16, 16), 32 => 16, stride = (2, 2), sigmoid, pad = SamePad()),
        BatchNorm(16), # 200, 200
        ConvTranspose((32, 32), 16 => 1, stride = (2, 2), sigmoid, pad = SamePad()),
        BatchNorm(1),
        Conv((1, 1), 1 => 1, pad = SamePad(), stride = 1),
        Conv((1, 1), 1 => 1, pad = SamePad(), stride = 1),
        Conv((1, 1), 1 => 1, pad = SamePad(), stride = 1),
        Conv((1, 1), 1 => 1, pad = SamePad(), stride = 1)
        #flatten # 40000 vector
    )
end