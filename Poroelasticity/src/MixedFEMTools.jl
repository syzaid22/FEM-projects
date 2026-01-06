function extract_component(component)
    return x -> x[component]
  end

# dimension-dependent 
function extract_row2d(row)
    return x -> VectorValue(x[1,row],x[2,row])
  end
  
function generate_model_unit_square(nk)
    domain =(0,1,0,1)
    n      = 2^nk
    partition = (n,n)
    model = CartesianDiscreteModel(domain, partition) |> simplexify
    model
end

function setup_model_labels_unit_square!(model)
    labels = get_face_labeling(model)
    add_tag!(labels,"Gamma_sig",[6,2,3,4,8]) #6top, right, and other 3 corners
    add_tag!(labels,"Gamma_u",[1,2,3,5,7]) # bottom and 7left and low-left corner
end  