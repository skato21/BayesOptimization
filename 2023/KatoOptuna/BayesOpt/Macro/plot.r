library(ggplot2)

theme_set(
    theme_bw() +
    theme(legend.position = "top")
)

df <- read.csv("ArchiveViewer-export-20230227-180316.csv", header=TRUE)
head(df)

df <- dplyr::filter(df, LIiBM.SP_16_5_1.ISNGL.KBP.1S > 3)

# Initiate a ggplot
b <- ggplot(df, aes(x = LIiMG.PX_R0_63.IWRITE.KBP+LIiMG.PX_R0_61.IWRITE.KBP, y = LIiMG.PY_R0_63.IWRITE.KBP+LIiMG.PY_R0_61.IWRITE.KBP))

b + geom_point(aes(color = LIiBM.SP_16_5_1.ISNGL.KBP.1S), size = 3) +
    scale_color_gradientn(colors = c("#00AFBB", "#E7B800", "#FC4E07"), name="e+ charge") +
    theme(legend.position = "right", aspect.ratio=1) +
    xlim(c(-4.5, -1.5)) + ylim(c(-3, 0))
