# Export the ClustMe Experiment 1 data (Abbas et al. 2019) to CSV files.
#
# Usage: Rscript tools/clustme_export.R <folder with the two .RData files> <output folder>
#
# Expects 1000_2gaussians_scatterplots_ClustMe_EXP1.RData (HumanData: point coordinates)
# and 1000_34answers_scatterplots_ClustMe_EXP1.RData (Answers: 34 judgments per scatterplot,
# 1 = "one cluster", 2 = "more than one cluster"). names(HumanData) match rownames(Answers).
# Writes <output>/NNNNN.csv (x;y) and <output>/NNNNN.answer.csv for scatterplots 1..1000;
# convert them to .npy with tools/convert_clustme.py.

args <- commandArgs(trailingOnly = TRUE)
input <- args[1]
output <- args[2]
dir.create(output, showWarnings = FALSE, recursive = TRUE)

load(file.path(input, "1000_2gaussians_scatterplots_ClustMe_EXP1.RData"))
load(file.path(input, "1000_34answers_scatterplots_ClustMe_EXP1.RData"))

for (t in seq_along(HumanData)) {
  name <- formatC(t, width = 5, format = "d", flag = "0")
  write.csv2(data.frame(HumanData[[t]]), file.path(output, paste0(name, ".csv")))
  write.csv2(Answers[t, ], file.path(output, paste0(name, ".answer.csv")))
}
