function basis_function(node, xi)
#      // Try the linear first. Just hard code the basis functions
#       double z = xi;
    value = 0
    if basisFunctionOrder ==1
      if node ==1
            value = (1 - xi)/2;
      elseif node ==2
            value = (1 + xi)/2;
       else
            println("Error: Your linear basis function should only have 2 nodes ")
        end
    else
       println("Only linear basis function defined right now") 
    end
    return value    
end

function basis_gradient(node, xi)
#      // Try the linear first. Just hard code the basis functions
#       double z = xi;
    value = 0
    if basisFunctionOrder ==1
      if node ==1
        value =-1/2
      elseif node ==2
          value =1/2
       else
        println("Error: Your linear basis function should only have 2 nodes ")
            end
    else
       println("Only linear basis function defined right now") 
    end
    return value    
end