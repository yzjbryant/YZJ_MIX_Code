#define XX      0           /* Defines for indexing in  */
#define YY      1           /* vectors          */
#define ZZ      2
#define DIM     3           /* Dimension of vectors     */
#define M_PI  3.1415926535897932384626433832795
#define DEG2RAD      (M_PI/180.0)

static inline void read_cryst(double box[DIM][DIM], double fa,
                              double fb, double fc, double alpha,
                              double beta, double gamma) 
{
  
  double cosa,cosb,cosg,sing;
  
    box[XX][XX] = fa;
    if ((alpha!=90.0) || (beta!=90.0) || (gamma!=90.0)) {
      if (alpha != 90.0) {
    cosa = cos(alpha*DEG2RAD);
      } else {
    cosa = 0;
      }
      if (beta != 90.0) {
    cosb = cos(beta*DEG2RAD);
      } else {
    cosb = 0;
      }
      if (gamma != 90.0) {
    cosg = cos(gamma*DEG2RAD);
    sing = sin(gamma*DEG2RAD);
      } else {
    cosg = 0;
    sing = 1;
      }
      box[YY][XX] = fb*cosg;
      box[YY][YY] = fb*sing;
      box[ZZ][XX] = fc*cosb;
      box[ZZ][YY] = fc*(cosa - cosb*cosg)/sing;
      box[ZZ][ZZ] = sqrt(fc*fc
             - box[ZZ][XX]*box[ZZ][XX] - box[ZZ][YY]*box[ZZ][YY]);
    } 
    else {
      box[YY][YY] = fb;
      box[ZZ][ZZ] = fc;
    }

}
