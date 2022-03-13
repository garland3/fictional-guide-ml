# Flowing this 
# https://www.youtube.com/watch?v=nMwjgCchTJc

using MLDatasets, Flux, Plots,  Statistics

using Images

train_x, train_y = MNIST.traindata(Float32)


test_x, test_y = MNIST.testdata(Float32)

size(train_x)

# l = @layout
# plots = 
# l = @layout [1;5]
# a = []
# for i in 1:5

idx = 14
x1 = train_x[:,:,idx];
y1 = train_y[idx]
println(y1)
colorview(Gray, x1', )

#     # heatmap(x1, c = :greys, title = y1)
#     push!(a, colorview(Gray, x1'))
# end
# # apply(plot,a)

# plot(a[1],a[2], layout = l)

xtrain = Flux.flatten(train_x);
xtest = Flux.flatten(test_x);
size(xtrain)

# ytrain, ytest = Flux.onehotbatch(train_y, 0:9)
ytrain, ytest = [Flux.onehotbatch(a, 0:9) for a in [train_y, test_y]]

# [println("$s and #n") for ]

# t = [(a*b) for  a = 0:5,  b =-5:3] 

(m, n, x)  = size(train_x)

?Flux.σ

function make_model()
    h = 60
    in_dims, out_dims = m*n, 10
    model = Flux.Chain(
        Dense(in_dims, h, Flux.σ), 
        Dense(h,h, Flux.σ),
        Dense(h, out_dims, Flux.σ)

    )
    return model
end
model = make_model()

loss(x,y) = Flux.Losses.mse(model(x), y)

# ?Flux.onecold

# Don't need the statisics dot notation. Just helsp to know where it comes from. 
accuracy(x,y) = Statistics.mean(Flux.onecold(model(x)) .==Flux.onecold(y)) 

opt = Descent(0.23)
# opt = Flux.ADAM(0.23)

data = [(xtrain, ytrain)];

parameters = Flux.params(model);

function say_status(xtrain, ytrain, state)
println("$state loss = $(loss(xtrain, ytrain))")
println("$state accuracy = $(accuracy(xtrain, ytrain))\n")
end

say_status(xtrain, ytrain, "Old")

Flux.train!(loss, parameters, data, opt)

say_status(xtrain, ytrain, "New")



# function SGD_Mnist(xtrain, ytrain, opt, epochs_max = 100_000)
grad_opt = Descent(0.23)

vector_length, num_imgs = size(xtrain)

# Make a new model so that we are staring from scratch
model = make_model()
loss(x,y) = Flux.Losses.mse(model(x), y)
accuracy(x,y) = Statistics.mean(Flux.onecold(model(x)) .==Flux.onecold(y)) 
parameters = Flux.params(model);
#     opt = 

function say_status_V2(xtrain, ytrain, state)
    println("$state loss = $(loss(xtrain, ytrain)) and accuracy = $(accuracy(xtrain, ytrain))")
end

epochs_max=10000
say_status_frequency =  Integer(round(epochs_max/10))
num_imgs_per_epoch = Integer(round(num_imgs/1000))  # get 0.1% of the data per epoch


# loop over `epochs_max` to train the model. 
println("Starting")
for epoch in 1:epochs_max

    # randomly select some data. 
    i = rand(1:num_imgs,num_imgs_per_epoch) 
   data = [(xtrain[:,i], ytrain[:,i])]    

    # Do the training, i.e. updat the weights
    Flux.train!(loss, parameters, data, opt)

    if epoch % say_status_frequency ==0
        say_status_V2(xtrain, ytrain, "Training at epoch  = $epoch, xtrain size $(size(xtrain))")
#             println("Epoch = $epoch")
#             @show loss(xtrain, ytrain)
#             @show accuracy(xtrain, ytrain)
    end
end
#     return model
# end

# m2 = SGD_Mnist(xtrain, ytrain, grad_opt)

# adam = Flux.ADAM(0.23)
# m3 = SGD_Mnist(xtrain, ytrain,  adam)

test1 = (x,y) -> x+y

test1(1,2)

@show test1(2,3)

Integer(1.0)

v = collect(1:Integer( round(5.6)))

i = rand(1:num_imgs-1)

_, num_imgs_test = size(xtest)


i = rand(1:num_imgs_test)
i

predict(i) = argmax(model(xtest[:,i]))-1

digit = predict(i)
actual =argmax(ytest[:,i])-1
println("pred = $digit, actual = $actual")

colorview(Gray, test_x[:,:,i]')

# size(ytest)

using TensorBoardLogger, Logging

logger = TBLogger("content/logs", tb_overwrite)

with_logger(logger) do 
   images = TBImage(train_x[:,:,1:10],WHN)
    @info "mnist/samples" pics = images log_step_increment = 0
end

#function to get dictionary of model parameters
function fill_param_dict!(dict, m, prefix)
    if m isa Chain
        for (i, layer) in enumerate(m.layers)
            fill_param_dict!(dict, layer, prefix*"layer_"*string(i)*"/"*string(layer)*"/")
        end
    else
        for fieldname in fieldnames(typeof(m))
            val = getfield(m, fieldname)
            if val isa AbstractArray
                val = vec(val)
            end
            dict[prefix*string(fieldname)] = val
        end
    end
end

#function to log information after every epoch
function TBCallback()
  param_dict = Dict{String, Any}()
  fill_param_dict!(param_dict, model, "")
  with_logger(logger) do
    @info "model" params=param_dict log_step_increment=0
    @info "train" loss=loss(xtrain, ytrain) acc=accuracy(xtrain, ytrain) log_step_increment=0
    @info "test" loss=loss(xtest, ytest) acc=accuracy(xtest, ytest)
  end
end

? Flux.ADAM

# function SGD_Mnist(xtrain, ytrain, opt, epochs_max = 100_000)
grad_opt =  Flux.ADAM(0.23)

vector_length, num_imgs = size(xtrain)

# Make a new model so that we are staring from scratch
model = make_model()
loss(x,y) = Flux.Losses.mse(model(x), y)
accuracy(x,y) = Statistics.mean(Flux.onecold(model(x)) .==Flux.onecold(y)) 
parameters = Flux.params(model);
#     opt = 





epochs_max=100
say_status_frequency =  Integer(round(epochs_max/10))
num_imgs_per_epoch = Integer(round(num_imgs/1000))  # get 0.1% of the data per epoch

m = model
l = loss
acc = accuracy
p = parameters
d  = data;

function say_status_V2( state)
    println("$state loss = $(loss(xtrain, ytrain)) and accuracy = $(accuracy(xtrain, ytrain))")
end

epochs_max=200
# it does seem much slower!!!!!


# loop over `epochs_max` to train the model. 
println("Starting")
for epoch in 1:epochs_max

    # randomly select some data. 
    i = rand(1:num_imgs,num_imgs_per_epoch) 
   data = [(xtrain[:,i], ytrain[:,i])]    

    # Do the training, i.e. updat the weights
    Flux.train!(l, p, d, opt, cb = Flux.throttle(TBCallback, 5))

    if epoch % say_status_frequency ==0
        say_status_V2(  "Training at epoch  = $epoch, xtrain size $(size(xtrain))")
#          println("$state loss = $(l(xtrain, ytrain)) and accuracy = $(acc(xtrain, ytrain))")
    end
end


epoch = "finished"
 say_status_V2(  "Training at epoch  = $epoch, xtrain size $(size(xtrain))")



