# Matrix-Matrix-Parallel
Matrix Matrix Multiplication using Serial, OpenMP, and CUDA


I thought this project was one of the more interesting things I have worked on
in my bachelor degree here. This project required KNOWLEDGE from the entire
semester. Having a strong knowledge on cache optimization, and how that alone can cause
speed up.

During this project I spent a lot more time on the code than the report at the end.
This in turn has led to a last minute submission (haha). I found running my code and
watching how fast it ran to be "exciting" as it was also passing the validation I
had implemented. This to me was convincing enough that the resulting matracies of
each strategy produce CORRECT solutions.

I have included the Spreadsheet that contains all of the data I ran in these expirements.

# Concluding Remarks
Parallelization is an incredibly important topic in computational sciences. The
exploitation of a single core in terms of locality is a good starting point to observe speed
up; however we can do better. Parallelization of a problem can in fact give performance
speed ups on unbelievable scales. How scheduling is implemented is another important
factor to keep in mind; in most cases, it is optimal to use the automated version (Guided
for OpenMP) as the API has a way of dynamically figuring what the best chunk size
should be for a problem; it is for this same reason, it is often a good idea to include
compiler flags that will optimize certain behaviours such as loop unrolling, and common
subexpression. Knowing the optimal number of threads based on the hardware of a
specific machine is a key factor; as allocating more threads than hardware threads
available, can lead to slower times, as the operating system at that point has to step in
switch who has time with each core. GPU computing give incredible results; the way the
hardware is designed is meant for number crunching. While it was observed that the
cores on a GPU were less efficient than a standard CPU, the overall quantity of cores
dwarfs the overall performance of the CPU. Tools such as OpenMP and CUDA are
incredibly important for large mathematical problems.