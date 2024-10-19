using Random
using StatsBase
using DataStructures
using Plots

# ------------------------- User Input Section -------------------------

# Time periods
const TIME_PERIODS = [1, 2, 3, 4]

# Define actions and their indices
const DECISIONS = ["IPC", "ACP", "RPPT", "IBI", "LBF", "LTP"]
const DECISION_INDICES = Dict(name => idx for (idx, name) in enumerate(DECISIONS))

# Initial modal shares (should sum to 1)
const initial_modal_shares = Dict("s" => 0.01, "c" => 0.50, "p" => 0.40, "o" => 0.09)

# Desired final biking modal shares
const final_states = [0.15, 0.10, 0.07]

# Action definitions
struct Action
    name::String
    effects::Dict{String, Float64}
    cooldown::Int
    one_time::Bool
    success_probability::Float64
    interactions::Dict{String, Float64}
    costs::Dict{String, Float64}
end

function Action(name; effects=Dict{String, Float64}(), cooldown=0, one_time=false, success_probability=1.0, interactions=Dict{String, Float64}(), costs=Dict{String, Float64}())
    effects = _normalize_effects(effects)
    return Action(name, effects, cooldown, one_time, success_probability, interactions, costs)
end

function _normalize_effects(effects::Dict{String, Float64})
    total_effect = sum(values(effects))
    if total_effect != 0
        unspecified_modes = setdiff(Set(["s", "c", "p", "o"]), Set(keys(effects)))
        adjustment = -total_effect / length(unspecified_modes)
        for mode in unspecified_modes
            effects[mode] = adjustment
        end
    end
    return effects
end

function evaluate_success_probability(action::Action, executed_actions)
    probability = action.success_probability
    for (interaction_action, adjustment) in action.interactions
        if get(executed_actions, interaction_action, 0) == 1
            probability += adjustment
        end
    end
    return clamp(probability, 0.0, 1.0)
end

function calculate_cost(action::Action, executed_actions)
    return action.costs["base"]
end

# Define the actions with their parameters
const ACTIONS = Dict(
    "IPC" => Action("IPC", cooldown=1, costs=Dict("base" => 0.01)),
    "ACP" => Action("ACP", effects=Dict("s" => 0.01, "c" => -0.005), cooldown=1, costs=Dict("base" => 0.01)),
    "RPPT" => Action("RPPT", effects=Dict("p" => -0.005, "s" => 0.0025, "c" => 0.0025), one_time=true, success_probability=0.2, interactions=Dict("IPC" => 0.2, "LBF" => 0.2), costs=Dict("base" => 0.1)),
    "IBI" => Action("IBI", effects=Dict("s" => 0.02, "c" => -0.01), one_time=true, success_probability=0.2, interactions=Dict("IPC" => 0.2, "LBF" => 0.2), costs=Dict("base" => 0.1)),
    "LBF" => Action("LBF", cooldown=1, success_probability=0.2, interactions=Dict("ACP" => 0.2), costs=Dict("base" => 0.01)),
    "LTP" => Action("LTP", effects=Dict("c" => -0.007, "s" => 0.0035, "p" => 0.0035), one_time=true, success_probability=0.2, interactions=Dict("ACP" => 0.2, "IPC" => 0.2), costs=Dict("base" => 0.01))
)

const MEAN_FIELD_EFFECTS = Dict(
    "base" => Dict("effects" => Dict("c" => 0.000, "p" => 0.000), "probability" => 0.8),
    "reactions" => Dict(
        "ACP" => Dict("effects" => Dict("c" => 0.002), "probability" => 0.7),
        "LTP" => Dict("effects" => Dict("c" => 0.002, "p" => 0.002), "probability" => 0.6),
        "RPPT" => Dict("effects" => Dict("c" => 0.0025, "s" => -0.0025), "probability" => 0.5)
    )
)

# Utility function (with normalized bike share gain)
function utility_function(path)
    if isempty(path)
        return -Inf  # Return negative infinity for empty paths
    end
    
    target_biking_share = minimum(final_states)
    initial_biking_share = initial_modal_shares["s"]
    max_possible_gain = target_biking_share - initial_biking_share
    
    total_utility = 0.0
    total_cost = 0.0
    
    for (i, step) in enumerate(path)
        modal_shares = step["modal_shares"]
        bike_share_gain = modal_shares["s"] - initial_biking_share
        normalized_gain = bike_share_gain / max_possible_gain
        step_utility = normalized_gain
        total_utility += step_utility

        # Calculate cost for this step
        step_cost = sum(get(ACTIONS[DECISIONS[i]].costs, "base", 0.0) for (i, d) in enumerate(step["decisions"]) if d == 1; init=0.0)
        total_cost += step_cost
    end
    
    # Normalize cost to be between 0 and 1
    max_possible_cost = sum(action.costs["base"] for action in values(ACTIONS)) * length(path)
    normalized_cost = total_cost / max_possible_cost
    
    # Combine normalized utility and cost (you can adjust the weights if needed)
    return 0.7 * total_utility - 0.3 * normalized_cost
end

# Exogenous factors (if any)
const EXOGENOUS_FACTORS = Dict(
    1 => Dict("effects" => Dict("s" => 0.01, "c" => -0.005, "p" => -0.0025, "o" => -0.0025), "probability" => 0.7, "label" => "E-Bikes"),
    2 => Dict("effects" => Dict("s" => 0.01, "c" => -0.005, "p" => -0.0025, "o" => -0.0025), "probability" => 0.01, "label" => "Pandemic"),
    3 => Dict("effects" => Dict("s" => 0.05, "c" => -0.025, "p" => -0.0125, "o" => -0.0125), "probability" => 0.05, "label" => "EU Biking Directive"),
    4 => Dict("effects" => Dict("s" => 0.01, "p" => -0.002, "c" => -0.004, "o" => -0.004), "probability" => 0.8, "label" => "Shared Biking"),
    5 => Dict("effects" => Dict("s" => 0.5, "p" => 0.3, "c" => -0.4, "o" => -0.4), "probability" => 1e-4, "label" => "Tipping point Social Perception")
)

# Add this function before the state_transition function

function mean_field_response(modal_shares, decisions)
    mf_effects = Dict("s" => 0.0, "c" => 0.0, "p" => 0.0, "o" => 0.0)
    
    # Apply base effects
    if rand() < MEAN_FIELD_EFFECTS["base"]["probability"]
        for (mode, delta) in MEAN_FIELD_EFFECTS["base"]["effects"]
            mf_effects[mode] += delta
        end
    end
    
    # Apply reaction effects based on our decisions
    for (action_name, reaction) in MEAN_FIELD_EFFECTS["reactions"]
        idx = DECISION_INDICES[action_name]
        if decisions[idx] == 1
            if rand() < reaction["probability"]
                for (mode, delta) in reaction["effects"]
                    mf_effects[mode] += delta
                end
            end
        end
    end
    
    return mf_effects
end

# Modify the state_transition function
function state_transition(modal_shares_prev, decisions, current_time)
    modal_shares = copy(modal_shares_prev)
    executed_actions = Dict(action_name => decisions[DECISION_INDICES[action_name]] for action_name in DECISIONS)
    actions_to_disable = Set()
    total_cost = 0.0
    debug_log = String[]

    for (action_name, action) in ACTIONS
        if get(executed_actions, action_name, 0) == 1
            success_prob = evaluate_success_probability(action, executed_actions)
            success = rand() <= success_prob
            if success
                push!(debug_log, "Action $action_name succeeded:")
                for (mode, delta) in action.effects
                    modal_shares[mode] += delta
                    push!(debug_log, "  $mode: $(round(delta, digits=4))")
                end
                total_cost += calculate_cost(action, executed_actions)
            else
                push!(debug_log, "Action $action_name failed")
                if action.one_time
                    push!(actions_to_disable, action_name)
                end
            end
        end
    end

    mf_effects = mean_field_response(modal_shares_prev, decisions)
    push!(debug_log, "Mean Field Effects:")
    for (mode, delta) in mf_effects
        modal_shares[mode] += delta
        push!(debug_log, "  $mode: $(round(delta, digits=4))")
    end

    # Exogenous factors
    exogenous_effects = get(EXOGENOUS_FACTORS, current_time, Dict())
    if !isempty(exogenous_effects)
        push!(debug_log, "Exogenous Effects ($(exogenous_effects["label"])):")
        if rand() < exogenous_effects["probability"]
            total_exogenous_effect = sum(values(exogenous_effects["effects"]))
            adjustment = -total_exogenous_effect / (length(modal_shares) - length(exogenous_effects["effects"]))
            for mode in keys(modal_shares)
                delta = get(exogenous_effects["effects"], mode, adjustment)
                modal_shares[mode] += delta
                push!(debug_log, "  $mode: $(round(delta, digits=4))")
            end
        else
            push!(debug_log, "  No $(exogenous_effects["label"]) effects applied (probability not met)")
        end
    end

    # Ensure modal shares are within [0,1]
    for mode in keys(modal_shares)
        if modal_shares[mode] < 0 || modal_shares[mode] > 1
            push!(debug_log, "Clipping $mode from $(round(modal_shares[mode], digits=4)) to $(round(clamp(modal_shares[mode], 0, 1), digits=4))")
        end
        modal_shares[mode] = clamp(modal_shares[mode], 0, 1)
    end

    # Normalize modal shares to sum to 1
    total_share = sum(values(modal_shares))
    if !isapprox(total_share, 1.0, atol=1e-6)
        push!(debug_log, "Adjusting modal shares to sum to 1:")
        adjustment_factor = 1 / total_share
        for mode in keys(modal_shares)
            old_share = modal_shares[mode]
            modal_shares[mode] *= adjustment_factor
            push!(debug_log, "  $mode: $(round(old_share, digits=4)) -> $(round(modal_shares[mode], digits=4))")
        end
    end

    return modal_shares, actions_to_disable, total_cost, debug_log
end
