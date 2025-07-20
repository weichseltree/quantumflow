using Colors, Plots

# Function to check if a point is in the Mandelbrot set
function mandelbrot(c, max_iter)
    z = c
    for n in 1:max_iter
        if abs2(z) > 4.0
            return n
        end
        z = z^2 + c * (-1)^n
    end
    return max_iter
end

# Function to generate the Mandelbrot set
function generate_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
    x = range(xmin, stop=xmax, length=width)
    y = range(ymin, stop=ymax, length=height)
    img = Array{RGB{Float64}, 2}(undef, length(x), length(y))

    for j in 1:length(y)
        for i in 1:length(x)
            c = Complex(x[i], y[j])
            m = mandelbrot(c, max_iter)
            img[i, j] = RGB(m/max_iter, 0, 1 - m/max_iter)
        end
    end

    return img
end

# Parameters
xmin, xmax = -1.0, 2.0
ymin, ymax = -1.5, 1.5
width, height = 300, 300
max_iter = 100

# Generate and plot the Mandelbrot set
mandelbrot_img = generate_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
plot(heatmap(mandelbrot_img'), axis=true, legend=false)
