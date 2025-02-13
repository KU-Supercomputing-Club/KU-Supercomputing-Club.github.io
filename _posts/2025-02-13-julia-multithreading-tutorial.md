---
title: A Quick Introduction to Parallel Programming with Julia 
description: A hands-on tutorial to parallel programming with multithreading in julia.
author: abir
date: 2025-02-13 15:00:00 +0000
categories: [Programming, Tutorial]
tags: [julia, multithreading, supercomputing, tutorial, programming]
pin: false
math: true
mermaid: true
image:
  path: https://ku-supercomputing-club.github.io/assets/img/commons/ksc_logo.png
  alt: ksc
---
### Example Julia code using multithreading
# **`test.jl`**
```julia
# julia -t 4 test.jl
using LinearAlgebra
using Base.Threads

function parallel_dot_race_condition(x,y)
    global dot_product=0
    #Blocking
	@threads for i in 1:length(x)
        #Race condition, undefined behavior
		global dot_product += x[i]*y[i]
	end
	return dot_product
end

function parallel_dot_locking(x,y)
    global dot_product=0
    l = ReentrantLock()
    #Blocking
	@threads for i in 1:length(x)
        #Locking, high syncronization overhead, not useful at all
        lock(l) do
		    global dot_product += x[i]*y[i]
        end
	end
	return dot_product
end

function parallel_dot_optimal(x,y)
    n = nthreads()
	partial_sums = zeros(n)
    #Blocking
	@threads for i in 1:length(x)
		partial_sums[threadid()] += x[i]*y[i]
	end
    dot_product=sum(partial_sums)
	return dot_product
end

len_vec=10^6
a=rand(len_vec)
b=rand(len_vec)

#=
Necessary to time the execution of multiple tests, 
as calling a function for the first time involves 
just-in-time (JIT) compilation
=#

println("\n\n\n\nSerial TESTS =======================")
println("\nFirst parallel test (includes compilation):")
println(@time dot(a,b))
println("\nSecond serial test (no compilation):")
println(@time dot(a,b))
println("\nThird serial test (no compilation):")
println(@time dot(a,b))

println("\n\n\nINCORRECT PARALLEL TESTS =======================")
println("\nFirst parallel test (includes compilation):")
println(@time parallel_dot_race_condition(a,b))
println("\nSecond parallel test (no compilation):")
println(@time parallel_dot_race_condition(a,b))
println("\nThird parallel test (no compilation):")
println(@time parallel_dot_race_condition(a,b))

println("\n\n\n\nSUBOPTIMAL PARALLEL TESTS =======================")
println("\nFirst parallel test (includes compilation):")
println(@time parallel_dot_locking(a,b))
println("\nSecond parallel test (no compilation):")
println(@time parallel_dot_locking(a,b))
println("\nThird parallel test (no compilation):")
println(@time parallel_dot_locking(a,b))

println("\n\n\n\nOPTIMAL PARALLEL TESTS =======================")
println("\nFirst parallel test (includes compilation):")
println(@time parallel_dot_optimal(a,b))
println("\nSecond parallel test (no compilation):")
println(@time parallel_dot_optimal(a,b))
println("\nThird parallel test (no compilation):")
println(@time parallel_dot_optimal(a,b))
```

If you want to execute this, type the following into your terminal:
```bash
# julia -t 16 test.jl
```
