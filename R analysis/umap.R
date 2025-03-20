library(Seurat)
library(ggplot2)
library(dplyr)

my_theme <- theme_bw(base_size = 14) +
  theme(panel.grid = element_blank(),
        panel.border = element_rect(color = "black"),
        axis.text = element_text(color = "black"),
        axis.title = element_text(face = "bold"))

data <- read.csv("umap.csv")

labels <- data$label
features <- data[, -1]

seurat_obj <- CreateSeuratObject(counts = t(as.matrix(features)), meta.data = data.frame(label = labels))

seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize", scale.factor = 10000)

seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2000)

seurat_obj <- ScaleData(seurat_obj)

seurat_obj <- RunPCA(seurat_obj, npcs = 30, verbose = FALSE)

seurat_obj <- RunUMAP(seurat_obj, dims = 1:10)

umap_df <- as.data.frame(Embeddings(seurat_obj, "umap"))
umap_df$label <- labels
my_theme <- theme_bw(base_size = 14) +
  theme(panel.grid.major = element_line(color = "gray80", linetype = "dashed"),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(color = "black"),
        axis.text = element_text(color = "black"),
        axis.title = element_text(face = "bold"))
pdf("umap.pdf", width = 8, height = 6)
ggplot(umap_df, aes(x = umap_1, y = umap_2, color = label)) +
  geom_point(size = 2, alpha = 0.8) +
  scale_color_manual(values = c("High nutrient" = "#F07673", "Low nutrient" = "#7998AD")) +
  stat_ellipse(aes(fill = label), type = "norm", alpha = 0.2, geom = "polygon", show.legend = FALSE) +
  labs(title = "UMAP of Microbiome Data",
       x = "UMAP 1",
       y = "UMAP 2",
       color = "Group") +
  my_theme
dev.off()
