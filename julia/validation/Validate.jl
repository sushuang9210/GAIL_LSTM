
# coding: utf-8

# # Run Validation

# In[1]:

include("validation.jl")
models = load_models()
modelnames = sort!(collect(keys(models)), rev=true)

# In[2]:

for model in modelnames
    validate(models[model], save=true, modelname=model)
    println(model)
end


# # Visualize Policy

# In[3]:

using AutoViz, Interact


# In[4]:

env_dict = Dict("trajdata_indeces" => [1],
                       "use_playback_reactive" => true,
                       "extract_core" => true,
                       "extract_temporal" => false,
                       "extract_well_behaved" => true,
                       "extract_neighbor_features" => false,
                       "extract_carlidar_rangerate" => true,
                       "carlidar_nbeams" => 20,
                       "roadlidar_nbeams" => 0,
                       "roadlidar_nlanes" => 2,
                       "nsteps" => 100,
                       "carlidar_max_range" => 100.0,
                       "roadlidar_max_range" => 100.0,
"model_all" => false)
simparams = Auto2D.gen_simparams(1, env_dict)


# In[7]:

# simparams.driver_model = Auto2D.load_gru_driver("./models/Dec11/policy_temp0_100.h5", 319);
simparams.driver_model = Auto2D.load_gru_driver("policy_gail.h5", 319);
model = models["bc_mlp"]
reset(simparams)
simstate = simparams.simstates[1]
trajdata = simparams.trajdatas[simstate.trajdata_index]
reset_hidden_state!(model)
empty!(simstate.rec)

if Symbol("gru") in fieldnames(model.net)
    model.net[:gru].h_prev = zeros(length(model.net[:gru].h_prev))
end
AutomotiveDrivingModels.observe!(model, simparams, simstate.scene, trajdata.roadway, simstate.egoid)
sim_rec = SceneRecord(200, 0.1)
for t = 1:200
    ego_action = rand(model)
    a = clamp(ego_action.a, -5.0, 3.0)
    ω = clamp(ego_action.ω, -0.1, 0.1)
    step(simparams, [a, ω])
    AutomotiveDrivingModels.observe!(model, simparams, simstate.scene, trajdata.roadway, simstate.egoid)
    update!(sim_rec, simstate.scene)
end


# In[8]:

scene = simparams.simstates[1].scene
egoid = simparams.simstates[1].egoid
trajdata = simparams.trajdatas[1]
@manipulate for pastframe in -length(sim_rec)+1 : 0
    render(get!(scene, sim_rec, pastframe), trajdata.roadway, cam=CarFollowCamera(egoid, 4.0),
       car_colors=Dict{Int,Colorant}(egoid=>COLOR_CAR_EGO))
end


# In[ ]:
