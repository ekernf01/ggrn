using HDF5
using JSON
using DataFrames
using CSV
using LinearAlgebra
using Statistics
using FilePathsBase

# TODO: implement reading from h5ad in Julia
function read_h5ad(filepath)
    h5open(filepath, "r") do file
    end
end

train = read_h5ad("from_to_docker/train.h5ad")

# Loading JSON files
perturbations = JSON.parsefile(FilePathsBase.joinpath("from_to_docker", "perturbations.json"))
kwargs = JSON.parsefile(FilePathsBase.joinpath("from_to_docker", "kwargs.json"))
ggrn_args = JSON.parsefile(FilePathsBase.joinpath("from_to_docker", "ggrn_args.json"))

# Set defaults
kwargs["pretrain_epochs"] = get(kwargs, "pretrain_epochs", "50")

# TODO: create two matched matrices as if from RNA velocity analysis
# https://rossinerbe.github.io/RNAForecaster.jl/dev/training/

# TODO: de-Pythonify this; currently it's been translated by chatGPT. 
predictions = DataFrame(perturbation = String[], expression_level_after_perturbation = Float64[])
for (i, goilevel) in enumerate(perturbations)
    goi, level = goilevel
    println("Predicting $goi")
    control = mean(train[train.obs["is_control"], goi].X) # Simplified, adjust based on actual data structure
    z = level > control ? 5 : -5
    mkpath("model")
    run(`prescient perturbation_analysis -i traindata.pt -p '$goi' -z $z --num_pcs 50 --model_path prescient_trained/kegg-growth-softplus_1_$(kwargs["pretrain_epochs"])-1e-06 --num_steps 10 --seed 2 -o experiments/`)
    println(readdir("experiments"))
    # Load results, adjust based on actual results format
    # result = load_result_somehow("experiments/result_file.pt")
    # Update predictions DataFrame
    # predictions = append!(predictions, DataFrame(...))
end

println("Saving results.")
# Adjust this part to write the predictions to an HDF5 file similar to how you'd use AnnData in Python
