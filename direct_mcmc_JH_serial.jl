# # MCMC inference using Julia Hub
# ### Load dependencies

# using Distributed
# using Pkg
# Pkg.activate(".")
# Pkg.instantiate()

# using JLD2, Dates, InlineStrings, Turing, LinearAlgebra, StatsBase, NamedArrays
# using CSV, DataFrames

# addprocs(3)


using JLD2, Dates, InlineStrings, Turing, LinearAlgebra, StatsBase, NamedArrays
using CSV, DataFrames



## ### Load data

onsets = load("data/onset_and_reported_cases.jld2")["onsets"]
onsets_hcw = load("data/onset_and_reported_cases.jld2")["onsets_hcw"]
reported = load("data/onset_and_reported_cases.jld2")["reported"]
reported_hcw = load("data/onset_and_reported_cases.jld2")["reported_hcw"]
dates = load("data/onset_and_reported_cases.jld2")["dates"]
case_district_names =
    load("data/onset_and_reported_cases.jld2")["case_district_names"] .|> String
dist_mat = load("data/named_dist_mat.jld2")["named_dist_mat"]
pop_data = CSV.File("data/uganda_district_pop_sizes.csv") |> DataFrame


## ### Gravity model set-up
# Basic gravity model prediction for where Mubende flux goes
@everywhere begin
    α̂, β̂ = [1.736715422247352, 2.4621672945356283]
    pops = pop_data.population_size
    flux = (pops .^ α̂) * (pops .^ α̂)' ./ (dist_mat .^ β̂)
    for i = 1:size(flux, 1)
        flux[i, i] = 0.0
    end

    # Set up matrix for probabiltiy of destination
    # dest_prob_mat_{ij} = probability from j → i

    dest_prob_mat = similar(flux)
    for j = 1:size(dest_prob_mat, 1)
        dest_prob_mat[:, j] = flux[j, :] / sum(flux[j, :]) #probability component
    end

    # Set-up the unnormalised movement rate by district
    move_prob = pops .^ (α̂ - 1) / maximum(pops .^ (α̂ - 1))
end
## Parameters
@everywhere begin
    γᵣ = 1 / 10 # Removal rate for undetecteds
    R_mean_prior = 0.588 * 5 + sum(0.588 * [zeros(7); [0.2 * exp(-γᵣ * t)  for t = 1:25]])
    w_ud = normalize([zeros(7); [exp(-γᵣ * t) for t = 1:30]], 1)
    rev_wud = reverse(w_ud)
end
## Generative model and likelihood in Turing


@info "Defining model"

@everywhere @model function ebola(
    onset_cases,
    reported_cases,
    onset_cases_hcw,
    reported_cases_hcw,
    dest_prob_mat,
    move_prob,
    rev_wud,
    prob_detect,
    obs_switch,
    onset_duration
)
    # Priors
    R₀ ~ Gamma(3, 4 / 3) #Basic R₀
    Rₕ ~ Exponential(1) # Mean number of healthcare workers infected
    move_scaler ~ Beta(1, 1) #scales the probability of moving away from district
    isolation_rate ~ Gamma(2, 0.25 / 2) #Rate at which infected people are found and isolated
    D_inf_duration ~ DiscreteUniform(3,10) #Duration of period eventually detected infecteds transmit for if not isolated

    # Set up arrays
    n_d = size(onset_cases, 1)
    n_t = size(onset_cases, 2)
    n = length(rev_wud)
    # reversed next generation distribution for detected cases
    rev_wd = [zeros(n - D_inf_duration - onset_duration); ones(D_inf_duration); zeros(onset_duration)] ./ D_inf_duration #Next generation distribution for detected cases
    # Arrays for infection events
    # Detected infections with time shift for reported cases
    detected_infections = (onset_cases + onset_cases_hcw) + [(reported_cases+reported_cases_hcw)[:, (D_inf_duration+1):end] zeros(Int64,n_d,D_inf_duration)]
    # Undetected infections treated as a random process
    undetected_infections = zeros(Int64, size(onset_cases))

    #Probability matrix for where infected people move to before onset (last dimension is any district with no reported cases)
    T =
        [Diagonal(1.0 .- move_scaler .* move_prob);zeros(n_d)'] +
        move_scaler .* dest_prob_mat .* move_prob' #Movement matrix

    # Initialise the unobserved infection process in district 1 (Mubende)
    for t = 1:2
        undetected_infections[1, t] ~ Poisson(detected_infections[1, t] * (1 - prob_detect) / (prob_detect) )
    end

    # Transmission dynamics and observation likelihood
    for t = 3:n_t
        τ = min(t - 1, n) # length of lookback
        lookback_times = (t-τ):(t-1)
        not_isolated_prob =
            lookback_times .|> t -> exp(-isolation_rate * max(t - obs_switch, 0.0))
        D = @view detected_infections[:, lookback_times]
        U = @view undetected_infections[:, lookback_times]
        _rev_wd = @view rev_wd[(n-τ+1):n]
        _rev_wud = @view rev_wud[(n-τ+1):n]

        #Chance of infection at district
        λ = R₀ * T * (D * (_rev_wd .* not_isolated_prob) + U * (_rev_wud))
        λ_hcw = Rₕ * (D * (_rev_wd .* not_isolated_prob) + U * (_rev_wud))
        #Generate infections and likelihood of their observation
        for d = 1:n_d
            #Generate unobserved infections
            undetected_infections[d, t] ~ Poisson(λ[d] * (1 - prob_detect))
            #Likelihood of observed cases
            if t < obs_switch
                onset_cases[d, t] ~ Poisson(λ[d] * prob_detect + 0.001)
                onset_cases_hcw[d, t] ~ Poisson(λ_hcw[d] + 0.01)
            end

            if t >= obs_switch - D_inf_duration && t <= n_t - D_inf_duration
                reported_cases[d, t + D_inf_duration] ~ Poisson(λ[d] * prob_detect)
                reported_cases_hcw[d, t + D_inf_duration] ~ Poisson(λ_hcw[d] + 0.01)
            end
        end

        #Accumulate the likelihood that no district outside has detected infections
        Turing.@addlogprob! logpdf(Poisson(λ[n_d+1] * prob_detect),0)

    end

    return undetected_infections, detected_infections
end



## Set up data
@everywhere begin
    _cases_dest_prob_mat = dest_prob_mat[case_district_names, case_district_names].array
    cases_dest_prob_mat = vcat(_cases_dest_prob_mat, 1.0 .- sum(_cases_dest_prob_mat,dims = 1))


    cases_move_prob = let
        idxs = [name ∈ case_district_names for name in pop_data.Districts]
        move_prob[idxs]
    end
end
##
@info "generating model"
@everywhere model = ebola(
    onsets[:,1:(end-6)],
    reported[:,1:(end-6)],
    onsets_hcw[:,1:(end-6)],
    reported_hcw[:,1:(end-6)],
    cases_dest_prob_mat,
    cases_move_prob,
    rev_wud,
    0.8,
    47 + 7,
    7
)

##

@everywhere sampler = Gibbs(
    MH(:R₀,:Rₕ, :move_scaler, :isolation_rate),
    PG(50, :undetected_infections, :D_inf_duration),
)
nsamples = 6000
nchains = 3
@info "Sampling"
# chain = sample(model, sampler, nsamples)
chain = sample(model, sampler, MCMCDistributed(), nsamples, nchains, progress=true)

CSV.write("mcmc/ebola_chain.csv", chain)

# ENV["RESULTS_FILE"] = "mcmc/ebola_chain.csv"