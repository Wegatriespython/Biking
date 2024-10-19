using Random
using StatsBase
using DataStructures
using Plots
using Combinatorics
using Measures

include("Backcasting_input.jl")

function set_random_seed(seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
end
# State node representation
mutable struct StateNode
    time::Int
    modal_shares::Dict{String, Float64}
    decisions::Union{Nothing, Vector{Int}}
    parent::Union{Nothing, StateNode}
    children::Vector{StateNode}
    cost::Float64
    debug_log::Vector{String}
    cooldowns::Dict{String, Int}
    disabled_actions::Set{String}
end

function StateNode(time, modal_shares; decisions=nothing, parent=nothing, cooldowns=nothing, disabled_actions=nothing, cost=0.0, debug_log=String[])
    if cooldowns === nothing
        cooldowns = Dict(action => 0 for action in DECISIONS)
    else
        cooldowns = copy(cooldowns)
    end
    if disabled_actions === nothing
        disabled_actions = Set{String}()
    else
        disabled_actions = copy(disabled_actions)
    end
    return StateNode(time, copy(modal_shares), decisions, parent, StateNode[], cost, debug_log, cooldowns, disabled_actions)
end

function build_graph(initial_modal_shares, final_states)
    root = StateNode(0, initial_modal_shares)
    queue = Deque{StateNode}()
    push!(queue, root)  # Use push! to add the root node to the queue
    paths = []

    while !isempty(queue)
        current_node = popfirst!(queue)
        if current_node.time == length(TIME_PERIODS)
            if any(isapprox(current_node.modal_shares["s"], final_state, atol=0.005) for final_state in final_states)
                path = []
                node = current_node
                while node.parent !== nothing
                    push!(path, Dict(
                        "time" => node.time,
                        "decisions" => node.decisions,
                        "modal_shares" => node.modal_shares,
                        "cost" => node.cost,
                        "debug_log" => node.debug_log
                    ))
                    node = node.parent
                end
                reverse!(path)
                push!(paths, path)
            end
            continue
        end

        # Decrease cooldowns
        cooldowns_next = Dict(action => max(current_node.cooldowns[action] - 1, 0) for action in DECISIONS)

        # Determine available actions
        available_actions = [action for action in DECISIONS if cooldowns_next[action] == 0 && !(action in current_node.disabled_actions)]

        # Generate all possible combinations of available actions (power set)
        action_combinations = [collect(comb) for r in 0:length(available_actions) for comb in combinations(available_actions, r)]

        # For each combination, create the decision vector
        for action_subset in action_combinations
            decisions = zeros(Int, length(DECISIONS))
            for action in action_subset
                idx = DECISION_INDICES[action]
                decisions[idx] = 1
            end

            modal_shares_next, actions_to_disable, step_cost, debug_log = state_transition(
                current_node.modal_shares,
                decisions,
                current_node.time
            )

            # Update cooldowns and disabled actions
            cooldowns_updated = copy(cooldowns_next)
            disabled_actions_updated = copy(current_node.disabled_actions)

            # Set cooldowns and disable actions as needed
            for action in action_subset
                action_obj = ACTIONS[action]
                idx = DECISION_INDICES[action]
                # Set cooldown if action has a cooldown
                if action_obj.cooldown !== nothing
                    cooldowns_updated[action] = action_obj.cooldown
                end
                # Disable one-time actions
                if action_obj.one_time
                    push!(disabled_actions_updated, action)
                end
            end

            # Disable actions that failed
            union!(disabled_actions_updated, actions_to_disable)

            # Create child node
            child_node = StateNode(
                current_node.time + 1,
                modal_shares_next,
                decisions=decisions,
                parent=current_node,
                cooldowns=cooldowns_updated,
                disabled_actions=disabled_actions_updated,
                cost=current_node.cost + step_cost,
                debug_log=debug_log
            )
            push!(current_node.children, child_node)
            push!(queue, child_node)
        end
    end

    return paths
end

function evaluate_path(path)
    return utility_function(path)
end

function find_best_path(paths)
    if isempty(paths)
        return nothing
    end
    return argmax(evaluate_path, paths)
end

function visualize_best_path(path)
    times = [0; [step["time"] for step in path]]
    shares = [initial_modal_shares["s"]; [step["modal_shares"]["s"] for step in path]]
    
    # Configure plot with adjusted margins and larger size
    p = plot(times, shares, marker=:circle,
             xlabel="Time Period", ylabel="Biking Modal Share", 
             title="Best Path for Increasing Biking Modal Share",
             label="Biking Modal Share Path", linewidth=2, 
             size=(1000, 800), left_margin=15mm, right_margin=20mm, top_margin=15mm, bottom_margin=15mm)

    # Loop through each step to add annotations
    for (i, step) in enumerate(path)
        y_pos = step["modal_shares"]["s"]
        
        annotations = String[]
        successful_actions = String[]
        failed_actions = String[]
        exogenous_effect = nothing
        
        for log_entry in step["debug_log"]
            if contains(log_entry, "succeeded")
                action = split(log_entry)[2]
                push!(successful_actions, action)
            elseif contains(log_entry, "failed")
                action = split(log_entry)[2]
                push!(failed_actions, action)
            elseif startswith(log_entry, "Exogenous Effects")
                exogenous_effect = split(log_entry, ":")[1]
            elseif contains(log_entry, "s:") && !isnothing(exogenous_effect)
                push!(annotations, exogenous_effect)
                exogenous_effect = nothing
            end
        end
        
        if !isempty(successful_actions)
            push!(annotations, "Successful: $(join(successful_actions, ", "))")
        end
        if !isempty(failed_actions)
            push!(annotations, "Failed: $(join(failed_actions, ", "))")
        end
        
        annotation_text = join(annotations, "\n")
        
        # Add annotation only if it exists, adjust position to avoid overlap
        if !isempty(annotation_text)
            annotate!(p, step["time"], y_pos + 0.005, text(annotation_text, 6))
        end
    end

    # Adding descriptive legend items for each action type
    legend_text = ["IPC: Improve Policy Coordination",
                   "ACP: Anti-Car Propaganda",
                   "RPPT: Raise Prices Public Transport",
                   "IBI: Invest in Bike Infrastructure",
                   "LBF: Lobby for Biking Finance",
                   "LTP: Levy Tax on Car Parking"]

    # Adding dummy series for the legend
    for label in legend_text
        plot!([NaN], [NaN], label=label, linestyle=:solid, marker=:none)
    end

    # Move legend to an outer position to avoid crowding
    plot!(p, legend=:outertopright)

    # Display and save the updated plot
    display(p)
    savefig(p, "best_path_plot.png")
    println("Plot saved as 'best_path_plot.png'")
end

function run_single_simulation()
    paths = build_graph(initial_modal_shares, final_states)
    return sort(paths, by=evaluate_path, rev=true)[1:min(10, length(paths))]
end

function monte_carlo_analysis(num_simulations::Int)
    best_paths = []
    for _ in 1:num_simulations
        simulation_best_paths = run_single_simulation()
        append!(best_paths, simulation_best_paths)
    end
    return best_paths
end

function save_best_paths(best_paths, filename::String)
    open(filename, "w") do io
        # Write header
        println(io, "Time,Decisions,Modal_Shares_s,Modal_Shares_c,Modal_Shares_p,Modal_Shares_o,Cost")
        
        for path in best_paths
            for step in path
                decision_names = join([DECISIONS[i] for (i, d) in enumerate(step["decisions"]) if d == 1], "|")
                modal_shares = step["modal_shares"]
                println(io, "$(step["time"]),$decision_names,$(modal_shares["s"]),$(modal_shares["c"]),$(modal_shares["p"]),$(modal_shares["o"]),$(round(step["cost"], digits=3))")
            end
            # Add a blank line between paths
            println(io)
        end
    end
end

function save_analysis_results(action_counts, total_costs, risk_categories, filename::String)
    open(filename, "w") do io
        println(io, "Category,Item,Value")
        
        for (action, count) in action_counts
            println(io, "Action Frequency,$action,$count")
        end
        
        println(io, "Cost Analysis,Average Cost,$(mean(total_costs))")
        println(io, "Cost Analysis,Max Cost,$(maximum(total_costs))")
        println(io, "Cost Analysis,Min Cost,$(minimum(total_costs))")
        
        for (category, count) in risk_categories
            println(io, "Risk Category,$category,$count")
        end
    end
end

function analyze_best_paths(best_paths)
    println("Total simulations: $(length(best_paths))")
    
    # Initialize data structures for analysis
    action_counts = Dict{String, Int}()
    total_costs = []
    risk_categories = Dict("High Risk" => 0, "Low Risk" => 0)
    
    for path in best_paths
        total_cost = 0.0
        for step in path
            # Count actions
            for (i, decision) in enumerate(step["decisions"])
                if decision == 1
                    action_name = DECISIONS[i]
                    action_counts[action_name] = get(action_counts, action_name, 0) + 1
                end
            end
            # Accumulate cost
            total_cost += step["cost"]
        end
        push!(total_costs, total_cost)
        
        # Categorize path risk
        if total_cost > 0.5  # Example threshold for high risk
            risk_categories["High Risk"] += 1
        else
            risk_categories["Low Risk"] += 1
        end
    end
    
    # Print analysis results
    println("Action Frequencies:")
    for (action, count) in action_counts
        println("  $action: $count")
    end
    
    println("\nCost Analysis:")
    println("  Average Cost: $(mean(total_costs))")
    println("  Max Cost: $(maximum(total_costs))")
    println("  Min Cost: $(minimum(total_costs))")
    
    println("\nRisk Categories:")
    for (category, count) in risk_categories
        println("  $category: $count")
    end

    # Save analysis results to a CSV file
    save_analysis_results(action_counts, total_costs, risk_categories, "analysis_results.csv")
end

function main()
    global paths
    set_random_seed()  # Remove the fixed seed

    num_simulations = 300
    best_paths = monte_carlo_analysis(num_simulations)
    
    # Save best paths to a CSV file
    save_best_paths(best_paths, "best_paths.csv")
    
    analyze_best_paths(best_paths)

    if isempty(best_paths)
        println("No feasible paths found.")
        return
    end

    best_path = find_best_path(best_paths)

    if best_path !== nothing
        println("Best Path:")
        for step in best_path
            decision_names = [DECISIONS[i] for (i, d) in enumerate(step["decisions"]) if d == 1]
            modal_shares_str = join([k * ": " * string(round(v, digits=3)) for (k, v) in step["modal_shares"]], ", ")
            println("Time $(step["time"]):")
            println("  Decisions: $(join(decision_names, ", "))")
            println("  Modal Shares: $modal_shares_str")
            println("  Cost: $(round(step["cost"], digits=3))")
            println("  Debug Log:")
            for log_entry in step["debug_log"]
                println("    $log_entry")
            end
            println()
        end
        visualize_best_path(best_path)
    else
        println("No feasible paths found.")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
