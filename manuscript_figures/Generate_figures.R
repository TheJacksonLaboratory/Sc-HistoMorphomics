library(openxlsx)
library(tidyverse)
library(ComplexHeatmap)
library(circlize)
library(ggpubr)

source_data <- paste0(
  res_dir,
  "Supplementary_Data_1.xlsx"
)

############################################################
# FIGURE 2A
############################################################

Fig2A <- read.xlsx(
  source_data,
  sheet = "Fig2A_correlation_matrix"
)

rownames(Fig2A) <- Fig2A$Feature
Fig2A$Feature <- NULL

cormat_Endo <- as.matrix(Fig2A)

############################################################
# FIGURE 2B
############################################################

Fig2B_dat <- read.xlsx(
  source_data,
  sheet = "Fig2B_boxplot"
)

p2B <- Fig2B_dat %>%
  ggplot(
    aes(
      x = calcs,
      y = abs(corrs)
    )
  ) +
  geom_violin(
    fill = "grey85",
    color = NA,
    scale = "width"
  ) +
  geom_boxplot(
    width = 0.15,
    linewidth = 0.25,
    outlier.size = 0.2
  ) +
  theme_classic(base_family = "Helvetica")

############################################################
# FIGURE 2C
############################################################

Fig2C <- read.xlsx(
  source_data,
  sheet = "Fig2C_cross_tissue"
)

rownames(Fig2C) <- Fig2C$compare
Fig2C$compare <- NULL

this_mat <- as.matrix(Fig2C)

############################################################
# FIGURE 3D
############################################################

Fig3D <- read.xlsx(
  source_data,
  sheet = "Fig3D_cluster_vs_celltype"
)

rownames(Fig3D) <- Fig3D$compare
Fig3D$compare <- NULL

thisMat_Fig3 <- as.matrix(Fig3D)

############################################################
# FIGURE 4D
############################################################

Fig4D <- read.xlsx(
  source_data,
  sheet = "Fig4D_cluster_vs_celltype"
)

rownames(Fig4D) <- Fig4D$compare
Fig4D$compare <- NULL

thisMat_Fig4 <- as.matrix(Fig4D)

############################################################
# FIGURE 5A
############################################################

Fig5A <- read.xlsx(
  source_data,
  sheet = "Fig5A_CCL5"
)

############################################################
# FIGURE 6
############################################################

Fig6 <- read.xlsx(
  source_data,
  sheet = "Fig6_CDKN1A"
)

############################################################
# FIGURE 7A
############################################################

Fig7A <- read.xlsx(
  source_data,
  sheet = "Fig7A_correlation_matrix"
)

corMat <- Fig7A

############################################################
# FIGURE 7B
############################################################

Fig7B_dat <- read.xlsx(
  source_data,
  sheet = "Fig7B_boxplot"
)

p7B <- Fig7B_dat %>%
  ggplot(
    aes(
      x = Category,
      y = Values
    )
  ) +
  geom_violin(
    fill = "grey85",
    color = NA,
    scale = "width"
  ) +
  geom_boxplot(
    width = 0.15,
    linewidth = 0.25,
    outlier.size = 0.2
  ) +
  theme_classic(base_family = "Helvetica")

############################################################
# FIGURE 7C
############################################################

Fig7C_dat <- read.xlsx(
  source_data,
  sheet = "Fig7C_SenMayo"
)

p7C <- Fig7C_dat %>%
  ggplot(
    aes(
      x = SenMayo,
      y = Texture_Correlation_Eosin_Mean
    )
  ) +
  geom_pointdensity(
    adjust = 0.005,
    size = 0.6
  ) +
  scale_color_viridis_c(
    option = "inferno",
    guide = "none"
  ) +
  stat_cor(
    method = "pearson",
    aes(label = ..r.label..)
  ) +
  theme_classic(base_family = "Helvetica")

############################################################
# FIGURE 8B
############################################################

Fig8B_dat <- read.xlsx(
  source_data,
  sheet = "Fig8B_RF_prediction"
)

p8B <- Fig8B_dat %>%
  ggplot(
    aes(
      x = pred,
      y = true
    )
  ) +
  geom_point(
    shape = 16,
    size = 0.5,
    alpha = 0.6,
    color = "black"
  ) +
  geom_smooth(
    method = "lm",
    se = FALSE,
    colour = "black",
    linewidth = 0.4
  ) +
  stat_cor(
    method = "pearson",
    aes(
      label = paste(
        ..r.label..,
        "\n",
        ..p.label..
      )
    )
  ) +
  theme_classic(base_family = "Helvetica")

############################################################
# FIGURE 8C
############################################################

Fig8C_dat <- read.xlsx(
  source_data,
  sheet = "Fig8C_RF_importance"
)

p8C <- Fig8C_dat %>%
  ggplot(
    aes(
      x = Feature,
      y = Mean,
      fill = category
    )
  ) +
  geom_col() +
  theme_classic(base_family = "Helvetica")