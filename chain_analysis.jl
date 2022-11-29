using DataFrames: _gen_colnames
using CSV, DataFrames, Turing, StatsPlots, MCMCChains

##


# mcmc_files = ["mcmc/ebola_chain1.csv", "mcmc/ebola_chain2.csv"]
mcmc_files = ["mcmc/ebola_chain_new.csv"]
chain_csv = mapreduce(fn -> DataFrame(CSV.File(fn)), vcat, mcmc_files)
# # relabel chains
# chain_csv = let
#     n = Integer(size(chain_csv, 1) / 2)
#     chain_csv.chain[(n+1):end] .+= 3
#     chain_csv
# end
##

colnames = names(chain_csv) |> x -> x[3:(end-1)] .|> Symbol
a = mapreduce(
    n -> Array(
        chain_csv[(chain_csv.chain.==n).&&(chain_csv.iteration.>300), :][:, 3:(end-1)],
    ),
    (x, y) -> cat(x, y, dims = 3),
    1:3,
)
chn = Chains(a, colnames)

##


plt_R0_chain =
    plot(chn[:R₀], lab = [j for i = 1:1, j = 1:6], alpha = 0.5, title = "R0 chain")

plt_Rh_chain =
    plot(chn[:Rₕ][:, :], lab = [j for i = 1:1, j = 1:6], alpha = 0.5, title = "R_H chain")

plt_move_chain = plot(
    chn[:move_scalar][:, :],
    lab = [j for i = 1:1, j = 1:6],
    alpha = 0.5,
    title = "move_scalar chain",
    legend = :bottomright,
)


plt_iso_chain = plot(
    chn[:isolation_rate][:, :],
    lab = [j for i = 1:1, j = 1:6],
    alpha = 0.5,
    title = "isolation rate chain",
)

##
include("model_definitions.jl")

gen_infs = generated_quantities(model, chn[1:5:end, :, [1, 2, 3]])
n_d, n_t = size(gen_infs[1][1])
##
# Stack gen_infs
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
