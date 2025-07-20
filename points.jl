using Distributions
using LinearAlgebra
using Plots

function generate_point_cloud(n::Int, dim::Int=2, std_dev::Float64=1.0)
    return randn(dim, n) * std_dev
end

# Generate point cloud
n = 5  # number of points
error_threshold=1e-6

points = eachcol(generate_point_cloud(n))
points = [[-0.5, 0.0], [0.5, 0.0], [0.1, 0.2], [-0.05, -0.2]]

# Function to calculate distances between all pairs of points
function calculate_distances(points)
    n = length(points)
    distances = Float64[]
    for i in 1:n
        for j in (i+1):n
            push!(distances, norm(points[i] - points[j]))
        end
    end
    return sort!(distances, rev=true)
end


# Function to find the closest existing distance to a given input value
function find_closest_distance(distances, target)
    return findmin((distances .- target).^2)
end

distances = calculate_distances(points)


#error, closest_index = find_closest_distance(distances, 2.6)


mutable struct Attempt
    error::Float64
    scale::Float64
    remaining_distances::Vector{Float64}
    remaining_indices::Vector{Int64}
    recovered_points::Vector{Vector{Float64}}
end

# Function to initialize the first two points
function new_attempt(distances)
    d = distances[1]
    return Attempt(
        0.0,
        d,
        distances[2:end] ./ d,
        collect(range(2, length(distances))),
        [[-0.5, 0.0], [0.5, 0.0]]
    )
end


function find_largest_pairs(values::Vector{Float64})
    n = length(values)
    sums = [values[1] + values[2]]  # First vector: current largest sums
    offsets = [1]  # Second vector: index offsets
    linked_list = [0]  # Third vector: linked list for sorting order
    largest_index = 1  # Pointer to current largest sum
    result = Tuple{Float64, Float64, Int64, Int64}[]

    while largest_index != 0 && sums[largest_index] >= 1.0
        i = largest_index
        j = i + offsets[i]
        
        # Add current largest pair to result
        push!(result, (values[i], values[j], i, j))
        
        # Update offset and calculate new sum
        offsets[i] += 1
        if i + offsets[i] <= n
            new_sum = values[i] + values[i + offsets[i]]
        else
            new_sum = 0.0
        end
        sums[i] = new_sum
        
        largest_index = linked_list[i]
        
        # Insert new sum into linked list
        prev = 0
        current = linked_list[i]
        while current != 0 && sums[current] >= new_sum
            prev = current
            current = linked_list[current]
        end
        if prev == 0
            largest_index = i
        else
            linked_list[prev] = i
        end
        linked_list[i] = current
        
        # Add new row if offset reaches 3 and we're not at the end
        if offsets[i] == 3 && i < n - 2
            push!(sums, values[i+1] + values[i+2])
            push!(offsets, 1)
            
            # Insert new index into linked list
            new_index = length(sums)
            prev = 0
            current = largest_index
            while current != 0 && sums[current] >= sums[new_index]
                prev = current
                current = linked_list[current]
            end
            if prev == 0
                largest_index = new_index
            else
                linked_list[prev] = new_index
            end
            push!(linked_list, current)
        end
    end
    
    return result
end


# Function to calculate the position of a third point given two points and distances
function calculate_third_point(d1, d2)
    x3 = (d1^2 - d2^2)/2
    y3 = sqrt(d1^2 - (x3 + 0.5)^2)
    return [x3, y3]
end


function check_new_point!(points, new_point, remaining_distances)
    for i in 1:length(points)
        new_distance = norm(points[i] - new_point)
        for j in 1:length(remaining_distances)
            println(remaining_distances[j] - new_distance)
        end
    end
    println(points)

    println(remaining_distances)
    
    push!(points, new_point)
end


# Function to recover points from distances
#function recover_points(distances, max_attempts=1000, error_threshold=1e-6)
attempts = [new_attempt(distances)]

#while !isempty(attempts) && length(attempts[1].recovered_points) < length(unique([d[2] for d in distances] ∪ [d[3] for d in distances]))
current_attempt = pop!(attempts)

pairs = find_largest_pairs(current_attempt.remaining_distances)

for (d1, d2, idx1, idx2) in pairs
    println("$d1, $d2, $idx1, $idx2")
    third_point = calculate_third_point(d1, d2)
    check_new_point!(current_attempt.recovered_points, third_point, current_attempt.remaining_distances)
    
end


"""

for (i, (d, p1_idx, p2_idx)) in enumerate(current_attempt.remaining_distances)
    if p1_idx <= length(current_attempt.recovered_points) && p2_idx <= length(current_attempt.recovered_points)
        p1 = current_attempt.recovered_points[p1_idx]
        p2 = current_attempt.recovered_points[p2_idx]
        new_point = calculate_third_point(p1, p2, d, norm(p1 - p2))
        
        new_attempt = Attempt(
            current_attempt.error,
            [current_attempt.remaining_distances[1:i-1]; current_attempt.remaining_distances[i+1:end]],
            [current_attempt.recovered_points; [new_point]]
        )
        
        # Calculate error
        for (d, i, j) in distances
            if i <= length(new_attempt.recovered_points) && j <= length(new_attempt.recovered_points)
                new_attempt.error += (norm(new_attempt.recovered_points[i] - new_attempt.recovered_points[j]) - d)^2
            end
        end
        
        if new_attempt.error < error_threshold
            return new_attempt.recovered_points
        end
        
        push!(attempts, new_attempt)
    end
end

sort!(attempts, by=a -> a.error)
if length(attempts) > max_attempts
    attempts = attempts[1:max_attempts]
end
#end

#return attempts[1].recovered_points
#end

"""

#recovered_points = recover_points(distances)


# Plot the points
scatter(points[1, :], points[2, :], 
        markersize=2, 
        legend=false, 
        title="2D Point Cloud")
xlabel!("X")
ylabel!("Y")
