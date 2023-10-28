library(GA)

t = function(c, a)
{
  (c*a + 2*c + 3*a)*sin(2*a+c)
}
c <- seq(-4,6,by=0.08)
a <- seq(3, 9, by=0.09)
z <- outer(c, a, t)
persp(c, a, z, theta = 30, phi = 30,
      expand = 0.5, col = 7)
image(c, a, z, col = terrain.colors(12))
contour(c, a, z, add = T, col = "gray50", nlevels = 12)

fm = function(obj){
  contour(c, a, z, col = "darkgray", nlevels = 12,
         main = paste("iteration =", obj@iter))
  points(obj@population, pch=20)
}

maxIter=500
rPopulacji=20

wynik.ga <-ga(type="real-valued",lower=c(-4, 3),upper=c(6, 9),
           fitness=function(x) t(x[1],x[2]),monitor=fm,
           popSize=20,pcrossover=0.85,pmutation=0.05,
           elitism=5,maxiter=60,seed=10)
summary(wynik.ga)
plot(wynik.ga)
