using CSV, DataFrames, Turing, StatsPlots, MCMCChains
using StatsPlots.PlotMeasures

##

##
mcmc_files = ["mcmc/ebola_chain_new1.csv", "mcmc/ebola_chain_new2.csv", "mcmc/ebola_chain_new3.csv"]


function gather_chains(mcmc_files)
    chain_csv = CSV.File(mcmc_files[1]) |> DataFrame
    if length(mcmc_files) > 1
        for (n, fn) = enumerate(mcmc_files[2:end])
            max_chn_number = maximum(chain_csv.chain) #Find how many independent chains have already been included
            _chain_csv = DataFrame(CSV.File(fn))
            _chain_csv.chain .+= max_chn_number #Increase the chain number
            chain_csv = vcat(chain_csv, _chain_csv) #combine the MCMC draws
        end
    end
    return chain_csv
end

chain_csv = gather_chains(mcmc_files)

## Construct a Chains object

colnames = names(chain_csv) |> x -> x[3:(end-1)] .|> Symbol
a = mapreduce(
    n -> Array(
        chain_csv[(chain_csv.chain .== n).&&(chain_csv.iteration .> 500), :][:, 3:(end-1)],
    ),
    (x, y) -> cat(x, y, dims = 3),
    1:maximum(chain_csv.chain),
)
chn = Chains(a, colnames)

##
include("model_definitions.jl")
##
parameter_chn = chn[:,1:5,1:end]
chn_plt = plot(parameter_chn,
                left_margin = 10mm,
                dpi = 250)
##
savefig(chn_plt, "plots/chain_plot.png")
crn_plt = corner(parameter_chn,
                    size = (1000,1000))
savefig(crn_plt, "plots/corner_plot.png")

dest_plt = heatmap(cases_dest_prob_mat[1:7,1:7],
                    xlabel = "From",
                    ylabel = "To",
                    xticks = (1:7, case_district_names[f]),
                    yticks = (1:7, case_district_names[f]))
bar(cases_move_prob)
##

gen_infs = generated_quantities(model, chn[1:10:end, :, :])
n_d, n_t = size(gen_infs[1][1])
##
# Stack the generated infections across chains
_gen_infs = gen_infs[:]

mean_detected_infs = [mean([inf[2][d, t] for inf in _gen_infs]) for d = 1:n_d, t = 1:n_t]
mean_undetected_infs = [mean([inf[1][d, t] for inf in _gen_infs]) for d = 1:n_d, t = 1:n_t]

lwr_detected_infs =
    mean_detected_infs .-
    [quantile([inf[2][d, t] for inf in _gen_infs], 0.025) for d = 1:n_d, t = 1:n_t]
upr_detected_infs =
    [quantile([inf[2][d, t] for inf in _gen_infs], 0.975) for d = 1:n_d, t = 1:n_t] .- mean_detected_infs

lwr_undetected_infs =
    mean_undetected_infs .-
    [quantile([inf[1][d, t] for inf in _gen_infs], 0.025) for d = 1:n_d, t = 1:n_t]
upr_undetected_infs =
    [quantile([inf[1][d, t] for inf in _gen_infs], 0.975) for d = 1:n_d, t = 1:n_t] .- mean_undetected_infs

##
plts = fill(plot(), 7)
for idx = 1:7
    plt = plot()

    scatter!(
        plt,
        dates[1:n_t] .- Day(7),
        mean_detected_infs[idx, 1:n_t],
        yerror = (lwr_detected_infs[idx, 1:n_t][:], upr_detected_infs[idx, 1:n_t][:]),
        lab = "D",
        markerstrokewidth = 3,
        markershape = :x,
        title = case_district_names[f][idx],
        legend = :topleft,
        alpha = 0.5,
        c = 1,
        lc = 4,
    )

    plot!(
        plt,
        dates[1:n_t] .- Day(7),
        mean_undetected_infs[idx, 1:n_t],
        ribbon = (lwr_undetected_infs[idx, 1:n_t], upr_undetected_infs[idx, 1:n_t]),
        lab = "U",
        lw = 5,
        title = case_district_names[f][idx],
        legend = :topleft,
        c = 2,
    )


    _onsets = (onsets[f,:][idx, :]+onsets_hcw[f,:][idx, :])[:]
    _reported = (reported[f,:][idx, :]+reported_hcw[f,:][idx, :])[:]

    if any(_onsets .> 0)
        scatter!(
            plt,
            dates[_onsets.>0],
            _onsets[_onsets.>0],
            lab = "Onsets",
            alpha = 0.5,
            c = 3,
        )
    end

    if any(_reported .> 0)
        scatter!(
            plt,
            dates[_reported.>0],
            _reported[_reported.>0],
            lab = "reported",
            alpha = 0.5,
            c = 4,
        )
    end

    plts[idx] = plt

end

##

districts = plot(
    plts[1],
    plts[2],
    plts[3],
    plts[4],
    plts[5],
    plts[6],
    plts[7],
    layout = (4, 2),
    size = (800, 1000),
    dpi = 250,
)

display(districts)

savefig(districts, "districts_plot.png")
##
# using CSV
# CSV.write("chain.csv", chain)

##
using Statistics

corr_mat = cor(chn[:, 1:4, [1, 2, 3, 4, 5, 6]])
_corr_mat = cor(chn[:, 1:4, [1, 2, 3, 4, 6]])
_corr_mat2 = cor(chn[:, 1:4, 5])
