using CSV
using DataFrames
using Random
using RNAForecaster
using Statistics
# include("./trainRNAForecaster.jl") # if we can't get the package to work

println("Reading metadata");
predictions_metadata = CSV.read("from_to_docker/predictions_metadata.csv", DataFrame) ;
training_metadata = CSV.read("from_to_docker/training_metadata.csv", DataFrame);
allGeneNames = CSV.read("from_to_docker/gene_metadata.csv", DataFrame)[:,1] |> Array{String, 1};

println("Reading expression data");
t0_matrix = CSV.read("from_to_docker/0.csv", DataFrame) |> Matrix{Float32};
t1_matrix = CSV.read("from_to_docker/1.csv", DataFrame) |> Matrix{Float32};
println("t0_matrix type: ", typeof(t0_matrix), ", size: ", size(t0_matrix));
println("t1_matrix type: ", typeof(t1_matrix), ", size: ", size(t1_matrix));

println("Training forecaster");
testForecaster = RNAForecaster.trainRNAForecaster(t0_matrix, t1_matrix, trainingProp=1.0);

println("Predicting perturbation outcomes");
predictions = Array{Float32, 2}(undef, size(t0_matrix, 1), size(predictions_metadata, 1));
predictions_metadata[!,"error_messages"] = fill("no error", nrow(predictions_metadata));
for cell in 1:nrow(predictions_metadata)

    # select correct starting cell type and timepoint
    timepoint = predictions_metadata[cell, "timepoint"];
    cell_type = predictions_metadata[cell, "cell_type"];
    subset_indices = findall((training_metadata[:, "timepoint"] .== timepoint) .& (training_metadata[:, "cell_type"] .== cell_type));
    starting_state = t0_matrix[:, subset_indices];

    try
        if predictions_metadata[cell, "is_control"]
            x = predictCellFutures(
                testForecaster[1], 
                starting_state, 
                Int64(predictions_metadata[cell, "prediction_timescale"])
            );
        else
            # Handle multi-gene perturbations: post-perturbation expression levels are comma-separated strings
            elap = predictions_metadata[cell, "expression_level_after_perturbation"];
            elap = split(string(elap), ',');
            elap = [parse(Float32, x) for x in elap];
        
            x = predictCellFutures(
                testForecaster[1], 
                starting_state, 
                Int64(predictions_metadata[cell, "prediction_timescale"]),
                perturbGenes = [String(s) for s in split(predictions_metadata[cell, "perturbation"], ",")], 
                geneNames = allGeneNames, 
                perturbationLevels = elap
            );
        end
        predictions[:, cell] = Statistics.mean(eachslice(x, dims=2))
    catch e
        predictions_metadata[cell, "error_messages"] = sprint(showerror, e);
        predictions[:, cell] .= NaN;
    end
end

println("Summary of errors:")
error_counts = combine(groupby(predictions_metadata, :error_messages), nrow)
println(error_counts)

println("Writing predictions")
CSV.write(
    "from_to_docker/predicted_expression.csv", 
    DataFrame(predictions, :auto), 
    writeheader=true, 
    bufsize = Int64(1E9),
);
CSV.write("from_to_docker/predictions_metadata.csv", predictions_metadata, writeheader=true)
