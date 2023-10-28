library(GA)

t = function(c, a)
{
  (c*a + 2*c + 3*a)*sin(2*a+c)
}
t = seq(-4,6,by=0.08)
a = seq(3, 9, by=0.09)
z = outer(c, a, t)
persp(c, a, z, theta = 30, phi = 30,
      expand = 0.5, col = 8)
image(c, a, z, col = terrain.colors(12))
contour(c, a, z, add = T, col = "magenta12", nlevels = 12)

fm = function(obj){
  curve(fp, -7,7,main=paste("iteration =", obj@iter))
  points(obj@population, obj@fitness, pch=20, col="red")
}

maxIter=500
rPopulacji=20

wynik.ga <- ga(type = "real-valued", fitness = fp, 
               lower = -7, upper = 7, popSize = rPopulacji,
               pcrossover = 0.9, pmutation=0.05,
               elitism = rPopulacji*0.1, monitor = fm,
               maxiter = maxIter, run = maxIter/10, 
               keepBest = TRUE, seed = 10)
summary(wynik.ga)
plot(wynik.ga)
