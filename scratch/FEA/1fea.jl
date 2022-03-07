u_0 = 0
u_l = 0.001
function f(x)
   return x*10e11 
end
num_elm = 100
h = 10e6
L = 0.1

f(2)

num_nodes = num_elm+1
step = L/num_elm
node_loc =  collect(range(0, length=num_nodes, stop=L))

elem_node_map  = zeros(Int8,num_elm,2); # Array{Float64, 2}(undef, 2, 3)


# elem_node_map  = zeros(Int8,num_elm,2); # Array{Float64, 2}(undef, 2, 3)
for i in collect(1:num_elm)
   elem_node_map[i,:]=[i,i+1] 
#     println(i)
#      setindex!(elem_node_map,
end

# elem_node_map

function get_dofs_for_elem(elm_num)
   # get th`ze  global dofs for this element
    return elem_node_map[elm_num,:]
    
end

get_dofs_for_elem(11)

for i in collect(1:num_elm)
    # loop over the elements
    
    dofs = get_dofs_for_elem(i)
    h_e  = node_loc[dofs[2]] - node_loc[dofs[1]]
#     - node_loc[dofs[0]]
    println("elm $i with h_e $h_e")
    
end



# ? collect


