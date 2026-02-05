function extract_component(component)
    return x -> x[component]
end

# 2D domain
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

# 3D domain
function extract_row3d(row)
    return x -> VectorValue(x[1,row],x[2,row],x[3,row])
end

function generate_model_unit_cube(nk)
    domain =(0,1,0,1,0,1)
    n      = 2^(nk-1) # starting at nk = 1 gives us no partitioning in first iteration; that is, apart from simplexification
    partition = (n,n,n)
    model = CartesianDiscreteModel(domain, partition) |> simplexify
    model
end

function setup_model_labels_unit_cube!(model)
    labels = get_face_labeling(model)
    add_tag!(labels,"Gamma_sig",[1,3,5,7,13,15,17,19,25])  # left face: corners (1,3,5,7); edges (13,15,17,19); face (25)
    add_tag!(labels,"Gamma_u",[2,4,6,8,9,10,11,12,13,14,16,18,20,21,22,23,24,26]) # every other corner, edge, face
end  