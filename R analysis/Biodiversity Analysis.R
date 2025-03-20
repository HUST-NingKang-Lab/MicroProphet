setwd("C:/Users/chudongliang/Desktop/invasion")
library(readr)
library(readxl)
library(tidyverse)
library(ggplot2)
library(ade4)   # 用于计算PcoA
library(vegan)# 用于计算距离
library(ggsci)
library(gridExtra)
library(patchwork)
library(ggsignif)
library(ggpubr)
ct = read_csv("rf_imputed.csv")
it = read_csv("invasion.csv")
ct$group = "RF"
it$group = "Actual"
pcoa_data <- rbind(ct, it)
any(is.na(pcoa_data))

df = pcoa_data[,3:14]
df <- as.data.frame(sapply(df, as.numeric))

df.dist <- vegdist(df,method='bray',na.rm=T)  

which(apply(df.dist, 1, function(x) any(is.na(x))))
df.dist[is.na(df.dist)] <- mean(df.dist, na.rm = TRUE)

color_map <- c("RF" = "#44757A", "Actual"= "#F07673")
pcoa <-  dudi.pco(df.dist,
                  scannf = F,   
                  nf=2 )         

data <- pcoa$li
data$name = rownames(data)
data$group = pcoa_data$group
data$group <- factor(data$group, levels =c("RF", "Actual"))
p<-ggplot(data,aes(x = A1,
                   y = A2,
                   color = group, fill = group))+
  geom_point(pch=20,size=1)+
  theme_classic()+
  geom_vline(xintercept = 0, color = 'gray', size = 0.4) +  
  geom_hline(yintercept = 0, color = 'gray', size = 0.4) +
  stat_ellipse(aes(x=A1,    
                   y=A2),
               geom = "polygon",
               level = 0.8,
               alpha=0.2)+
  stat_ellipse(level = 0.95,lty=2,size=0.8) +
  scale_color_manual(values = color_map) + 
  scale_fill_manual(values = color_map) +  
 
  labs(  )
    x = paste0("PCoA1 (",as.character(round(pcoa$eig[1] / sum(pcoa$eig) * 100,2)),"%)"),
    y = paste0("PCoA2 (",as.character(round(pcoa$eig[2] / sum(pcoa$eig) * 100,2)),"%)")
  )+
  theme(legend.position = "bottom")

boxplot_x <- ggplot(data, aes(x = group, y = A1, fill = group)) +
  geom_boxplot() +
  theme_classic() +
  scale_fill_manual(values = color_map) + 
  labs(y = "PCoA1", x = "Group") +
  coord_flip() +
  theme(legend.position = "none")+
  stat_compare_means(comparisons = list(c("RF", "Actual")), 
                     label = "p.signif")

boxplot_y <- ggplot(data, aes(x = group, y = A2, fill = group)) +
  geom_boxplot() +
  theme_classic() +
  labs(y = "PCoA2", x = "Group") +
  scale_fill_manual(values = color_map) + 
  theme(legend.position = "none")+
  stat_compare_means(comparisons = list(c("RF", "Actual")), 
                     label = "p.signif")

pcoa_combined <- (boxplot_x / p) + 
  plot_layout(heights = c(1, 3))  

combined_plot <- (pcoa_combined | boxplot_y) + 
  plot_layout(widths = c(3, 1), heights = c(3, 3))  
combined_plot
ggsave("rf_true.pdf",combined_plot,height = 6,width = 6, dpi = 300)

df = pcoa_data[,3:14]
df <- as.data.frame(sapply(df, as.numeric))

shannon_diversity <- diversity(df, index = "shannon")  
simpson_diversity <- diversity(df, index = "simpson")  

alpha_df <- data.frame(
  Sample = rownames(df), 
  Shannon = shannon_diversity,  
  Simpson = simpson_diversity,  
  Group = pcoa_data$group  
)

alpha = ggplot(alpha_df, aes(x = Group, y = Shannon, fill = Group)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Shannon Alpha Diversity", y = "Shannon Diversity", x = "Group") +
  scale_fill_manual(values = color_map) + 
  theme(legend.position = "none")+
  stat_compare_means(comparisons = list(c("RF", "Actual")), 
                     label = "p.signif")
print(alpha)
ggsave("shannon.pdf",alpha,height = 6,width = 6, dpi = 300)

