 begin
    using JLD2, Dates, InlineStrings, Turing, LinearAlgebra, StatsBase, NamedArrays
    using CSV, DataFrames, Downloads, AdvancedMH
end

@info "Load data from remote"

## ### Load data
 begin
    url_pop_data = "https://warwick.ac.uk/fac/cross_fac/zeeman_institute/staffv2/sam_brand/open_data/uganda_district_pop_sizes.csv"
    url_dist_mat = "https://warwick.ac.uk/fac/cross_fac/zeeman_institute/staffv2/sam_brand/open_data/named_dist_mat.jld2"
    url_case_data = "https://warwick.ac.uk/fac/cross_fac/zeeman_institute/staffv2/sam_brand/open_data/onset_and_reported_cases.jld2"

    case_data_dict = load(Downloads.download(url_case_data))
    onsets = case_data_dict["onsets"]
    onsets_hcw = case_data_dict["onsets_hcw"]
    reported = case_data_dict["reported"]
    reported_hcw = case_data_dict["reported_hcw"]
    dates = case_data_dict["dates"]
    case_district_names = case_data_dict["case_district_names"] .|> String
    named_dist_mat = load(Downloads.download(url_dist_mat))["named_dist_mat"]
    dist_mat = load(Downloads.download(url_dist_mat))["dist_mat"]
    pop_data = CSV.File(Downloads.download(url_pop_data)) |> DataFrame
end



@model function ebola(
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
    move_scalar ~ Beta(1, 1) #scales the probability of moving away from district
    isolation_rate ~ Gamma(2, 0.25 / 2) #Rate at which infected people are found and isolated
    D_inf_duration ~ DiscreteUniform(3,7) #Duration of period eventually detected infecteds transmit for if not isolated

    # Set up arrays
    n_d = size(onset_cases, 1)
    n_t = size(onset_cases, 2)
    n = length(rev_wud)
    # reversed next generation distribution for detected cases
    rev_wd = [zeros(n - D_inf_duration - onset_duration); ones(D_inf_duration); zeros(onset_duration)] ./ D_inf_duration #Next generation distribution for detected cases
    # Arrays for infection events
    # Detected infections with time shift for reported cases
    detected_infections = (onset_cases + onset_cases_hcw) + [(reported_cases + reported_cases_hcw)[:, (D_inf_duration+1):end] zeros(Int64,n_d,D_inf_duration)]
    # Undetected infections treated as a random process
    undetected_infections = zeros(Int64, size(onset_cases))

    #Probability matrix for where infected people move to before onset (last dimension is any district with no reported cases)
    T = [Diagonal(1.0 .- move_scalar .* move_prob);zeros(n_d)'] + move_scalar .* dest_prob_mat .* move_prob' #Movement matrix

    # Initialise the unobserved infection process in district 1 (Mubende)
    for t = 1:2
        undetected_infections[1, t] ~ Poisson(detected_infections[1, t] * (1 - prob_detect) / (prob_detect) )
    end

    # Transmission dynamics and observation likelihood
    for t = 3:n_t
        τ = min(t - 1, n) # length of lookback
        lookback_times = collect((t-τ):(t-1)) # time points looking back over
        not_isolated_prob =
            lookback_times |> reverse .|> ts -> exp(-isolation_rate * min(ts - max(obs_switch,lookback_times[1]), 0.0)) #reverse order probability of not-having been isolated
        D = @view detected_infections[:, lookback_times]
        U = @view undetected_infections[:, lookback_times]
        _rev_wd = @view rev_wd[(n-τ+1):n]
        _rev_wud = @view rev_wud[(n-τ+1):n]

        #Chance of infection at district
        λ = R₀ * T * (D * (_rev_wd .* not_isolated_prob) + U * _rev_wud)
        λ_hcw = Rₕ * (D * (_rev_wd .* not_isolated_prob) + U * _rev_wud)
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



@info "generating model"

## Set up data
begin
    cases_dest_prob_mat = let
        idxs = [name ∈ case_district_names for name in pop_data.Districts]
        f = findall(sum(onsets + reported, dims = 2)[:] .> 0) # Remove districts with no non-HCW cases
        _cases_dest_prob_mat = dest_prob_mat[idxs, idxs][f,f]
        cases_dest_prob_mat = vcat(_cases_dest_prob_mat, 1.0 .- sum(_cases_dest_prob_mat,dims = 1))
    end
    
    cases_move_prob = let
        idxs = [name ∈ case_district_names for name in pop_data.Districts]
        f = findall(sum(onsets + reported, dims = 2)[:] .> 0) # Remove districts with no non-HCW cases
        move_prob[idxs][f]
    end
end
##
f = findall(sum(onsets + reported, dims = 2)[:] .> 0) # Remove districts with no non-HCW cases

model = ebola(
    onsets[f,:],
    reported[f,:],
    onsets_hcw[f,:],
    reported_hcw[f,:],
    cases_dest_prob_mat,
    cases_move_prob,
    rev_wud,
    0.8,
    47 + 7,# Detection time in terms of the 7 day lagged onset times
    7
)
