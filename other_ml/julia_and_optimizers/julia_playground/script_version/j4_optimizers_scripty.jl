include("j3_optimization.jl")

@time begin
    n = 300
    update_fre = 20
    for i in range(1,length = n)     
        xpoints, ypoints = get_data(false);
        f = make_inter_function(xpoints,ypoints);
        x, xs = find_a_min_grad(f);
        x, xs = find_a_min_momentum(f);
        x, xs = find_a_min_rmsp(f);
        x, xs = find_a_min_adam(f);
        if mod(i,update_fre)==0
           println("-----------------------------------------------")
           println("On iteration $i") 
           println("-----------------------------------------------")
        end

    end

end