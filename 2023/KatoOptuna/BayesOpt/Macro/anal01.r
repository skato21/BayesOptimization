library(ggplot2)

theme_set(
    theme_bw() +
    theme(legend.position = "top")
)

df <- read.csv("ArchiveViewer-export-20230227-180316.csv", header=TRUE)
head(df)

# ArchiveViewer-export-20230227-180454.csv
#df <- dplyr::filter(df, LIiBM.SP_16_5_1.ISNGL.KBP.1S > 3, LIiMG.PX_R0_63.IWRITE.KBP+LIiMG.PX_R0_61.IWRITE.KBP > -2.6, LIiMG.PX_R0_63.IWRITE.KBP+LIiMG.PX_R0_61.IWRITE.KBP < -2.2, LIiMG.PY_R0_63.IWRITE.KBP+LIiMG.PY_R0_61.IWRITE.KBP > -2.2, LIiMG.PY_R0_63.IWRITE.KBP+LIiMG.PY_R0_61.IWRITE.KBP < -1.2)

# ArchiveViewer-export-20230227-180316.csv
df <- dplyr::filter(df, LIiBM.SP_16_5_1.ISNGL.KBP.1S > 3, LIiMG.PX_R0_63.IWRITE.KBP+LIiMG.PX_R0_61.IWRITE.KBP > -3.5, LIiMG.PX_R0_63.IWRITE.KBP+LIiMG.PX_R0_61.IWRITE.KBP < -2.5, LIiMG.PY_R0_63.IWRITE.KBP+LIiMG.PY_R0_61.IWRITE.KBP > -2.0, LIiMG.PY_R0_63.IWRITE.KBP+LIiMG.PY_R0_61.IWRITE.KBP < -1.)

# plot 1
b1 <- ggplot(df, aes(x = LIiMG.PX_R0_63.IWRITE.KBP, y = LIiMG.PX_R0_61.IWRITE.KBP))
#b1 <- ggplot(df, aes(x = LIiMG.PY_R0_63.IWRITE.KBP, y = LIiMG.PY_R0_61.IWRITE.KBP))
b1 + geom_point(aes(color = LIiBM.SP_16_5_1.ISNGL.KBP.1S), size = 3) +
    scale_color_gradientn(colors = c("#00AFBB", "#E7B800", "#FC4E07"), name="e+ charge") +
    theme(legend.position = "right", aspect.ratio=1)
#+ ylim(c(-3.5, -2.5)) + xlim(c(0.0, 1.0)) 
