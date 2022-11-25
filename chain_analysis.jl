using CSV, DataFrames, Turing, StatsPlots

##

chain_csv = CSV.File("mcmc/ebola_chain.csv") |> DataFrame

##

chain1 = chain_csv[(chain_csv.chain .== 1) .&& (chain_csv.iteration .> 1000),:]
a = cha
