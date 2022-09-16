"""
Testing all kinds



"""

begin
    using Flux
    using NNlib

    using Flux: Chain
end


"""
tn_2a (generic function with 1 method)

Chain(
  Conv((21, 21), 3 => 8, relu, pad=10),  # 10_592 parameters
  BatchNorm(8),                         # 16 parameters, plus 16
  Conv((21, 21), 8 => 8, relu, pad=10),  # 28_232 parameters
  BatchNorm(8),                         # 16 parameters, plus 16
  Conv((21, 21), 8 => 8, relu, pad=10),  # 28_232 parameters
  BatchNorm(8),                         # 16 parameters, plus 16
  Conv((21, 21), 8 => 8, relu, pad=10),  # 28_232 parameters
  BatchNorm(8),                         # 16 parameters, plus 16
  Conv((21, 21), 8 => 8, relu, pad=10),  # 28_232 parameters
  BatchNorm(8),                         # 16 parameters, plus 16
  Conv((21, 21), 8 => 8, relu, pad=10),  # 28_232 parameters
  BatchNorm(8),                         # 16 parameters, plus 16
  Conv((21, 21), 8 => 8, relu, pad=10),  # 28_232 parameters
  BatchNorm(8),                         # 16 parameters, plus 16
  Conv((21, 21), 8 => 8, relu, pad=10),  # 28_232 parameters
  BatchNorm(8),                         # 16 parameters, plus 16
  Conv((17, 17), 8 => 16, relu, pad=8, stride=2),  # 37_008 parameters
  BatchNorm(16),                        # 32 parameters, plus 32
  Conv((17, 17), 16 => 16, relu, pad=8),  # 74_000 parameters
  BatchNorm(16),                        # 32 parameters, plus 32
  Conv((17, 17), 16 => 16, relu, pad=8),  # 74_000 parameters
  BatchNorm(16),                        # 32 parameters, plus 32
  Conv((17, 17), 16 => 16, relu, pad=8),  # 74_000 parameters
  BatchNorm(16),                        # 32 parameters, plus 32
  Conv((17, 17), 16 => 16, relu, pad=8),  # 74_000 parameters
  BatchNorm(16),                        # 32 parameters, plus 32
  Conv((17, 17), 16 => 16, relu, pad=8),  # 74_000 parameters
  BatchNorm(16),                        # 32 parameters, plus 32
  Conv((17, 17), 16 => 16, relu, pad=8),  # 74_000 parameters
  BatchNorm(16),                        # 32 parameters, plus 32
  Conv((17, 17), 16 => 16, relu, pad=8),  # 74_000 parameters
  BatchNorm(16),                        # 32 parameters, plus 32
  Conv((13, 13), 16 => 32, relu, pad=6, stride=2),  # 86_560 parameters
  BatchNorm(32),                        # 64 parameters, plus 64
  Conv((13, 13), 32 => 32, relu, pad=6),  # 173_088 parameters
  BatchNorm(32),                        # 64 parameters, plus 64
  Conv((13, 13), 32 => 32, relu, pad=6),  # 173_088 parameters
  BatchNorm(32),                        # 64 parameters, plus 64
  Conv((13, 13), 32 => 32, relu, pad=6),  # 173_088 parameters
  BatchNorm(32),                        # 64 parameters, plus 64
  Conv((13, 13), 32 => 32, relu, pad=6),  # 173_088 parameters
  BatchNorm(32),                        # 64 parameters, plus 64
  Conv((13, 13), 32 => 32, relu, pad=6),  # 173_088 parameters
  BatchNorm(32),                        # 64 parameters, plus 64
  Conv((13, 13), 32 => 32, relu, pad=6),  # 173_088 parameters
  BatchNorm(32),                        # 64 parameters, plus 64
  Conv((13, 13), 32 => 32, relu, pad=6),  # 173_088 parameters
  BatchNorm(32),                        # 64 parameters, plus 64
  Conv((9, 9), 32 => 64, relu, pad=4, stride=3),  # 165_952 parameters
  BatchNorm(64),                        # 128 parameters, plus 128
  Conv((9, 9), 64 => 64, relu, pad=4),  # 331_840 parameters
  BatchNorm(64),                        # 128 parameters, plus 128
  Conv((9, 9), 64 => 64, relu, pad=4),  # 331_840 parameters
  BatchNorm(64),                        # 128 parameters, plus 128
  Conv((9, 9), 64 => 64, relu, pad=4),  # 331_840 parameters
  BatchNorm(64),                        # 128 parameters, plus 128
  ConvTranspose((8, 8), 64 => 32, σ, pad=3, stride=2),  # 131_104 parameters
  BatchNorm(32),                        # 64 parameters, plus 64
  ConvTranspose((16, 16), 32 => 16, σ, pad=7, stride=2),  # 131_088 parameters
  BatchNorm(16),                        # 32 parameters, plus 32
  ConvTranspose((32, 32), 16 => 1, σ, pad=15, stride=2),  # 16_385 parameters
  BatchNorm(1),                         # 2 parameters, plus 2
  Conv((1, 1), 1 => 1),                 # 2 parameters
  Conv((1, 1), 1 => 1),                 # 2 parameters
  Conv((1, 1), 1 => 1),                 # 2 parameters
  Conv((1, 1), 1 => 1),                 # 2 parameters
  Base.Iterators.flatten,
  collect,
  Dense(36864, 9216),                   # 339_747_840 parameters
)         # Total: 134 trainable arrays, 343_250_803 parameters,
          # plus 62 non-trainable, 1_506 parameters, summarysize 1.279 GiB.
"""
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


model = tn_2a()


begin
    test = rand(Float32, 288, 288, 3, 1)
    pred1 = model(test)
end

length(pred1)

model

o1 = ConvTranspose((8, 8), 64 => 32, stride = (2, 2), relu, pad = SamePad())(test2)
o2 = ConvTranspose((16, 16), 32 => 16, stride = (2, 2), relu, pad = SamePad())(o1)
o3 = ConvTranspose((32, 32), 16 => 1, stride = (2, 2), relu, pad = SamePad())(o2)

o4 = o3 |> flatten

typeof(o4)

out2 = ConvTranspose((5, 5), 3 => 7, stride = 3, pad = SamePad())(test)

###
test2 = rand(Float32, 300, 300, 3, 1);

model = tn_1d()
out = model(test2);
println(size(out))

th(x) = sigmoid(20 * (x - 0.5))

maximum(o4)
maximum(o4 |> th)

20 * o4

test3 = rand(Float32, 20, 20, 1, 1)

test3 .|> th

test3 .|> sigmoid

maximum(test3 |> th)

Conv((1, 1), 1 => 1, sigmoid, pad = SamePad(), stride = 1)(test3)


count(x -> (x > 1.9 && x <= 2.0) || (x > 0.9 && x <= 1.0), test3)

test4 = rand(Float32, 20, 20, 1, 1)

test5 = test3 + test4

count(x -> (x > 1.9 && x <= 2.0) || (x > 0.9 && x <= 1.0), test5)

test6 = replace(x -> (x > 0.9 && x <= 1.0) ? 1.0 : x, test4)

count(x -> x == 1.0, test6)