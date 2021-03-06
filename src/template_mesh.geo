jet_positions[] = {1.5707963267948966, 4.71238898038469};  // 90, 270
DefineConstant[
length = {2.2, Name "Channel length"}
front_distance = {0.2, Name "Cylinder center distance to inlet"}
bottom_distance = {0.2, Name "Cylinder center distance from bottom"}
jet_radius = {0.05, Name "Cylinder radius"}
jet_width = {10*Pi/180, Name "Jet width in radians"}
width = {0.41, Name "Channel width"}
// Mesh size specs
cylinder_bl_width = {0.075, Name "Thickness of fine region close to cyliner"}
wake_length = {1.0, Name "Length of fine box region behing cylinder"}
cylinder_inner_size = {3E-3, Name "Mesh size close to cylinder"}
cylinder_outer_size = {7E-3, Name "Mesh size at cylinders's disk"}
wake_size = {7E-3, Name "Mesh size in the wake"}g
outlet_size = {1E-1, Name "Mesh size at outlet"}
inlet_size = {2E-2, Name "Mesh size at inlet"}
];

// Seed the cylinder
center = newp;
Point(center) = {0, 0, 0, cylinder_inner_size};

n = #jet_positions[];

radius = jet_radius;

If(n > 0)
  cylinder[] = {};
  lower_bound[] = {};
  uppper_bound[] = {};

  //  Define jet surfaces
  For i In {0:(n-1)}

      angle = jet_positions[i];
  
      x = radius*Cos(angle-jet_width/2);
      y = radius*Sin(angle-jet_width/2);
      p = newp;
      Point(p) = {x, y, 0, cylinder_inner_size};
      lower_bound[] += {p};

      x0 = radius*Cos(angle);
      y0 = radius*Sin(angle);
      arch_center = newp;
      Point(arch_center) = {x0, y0, 0, cylinder_inner_size};

      x = radius*Cos(angle+jet_width/2);
      y = radius*Sin(angle+jet_width/2);
      q = newp;
      Point(q) = {x, y, 0, cylinder_inner_size};
      upper_bound[] += {q};
  
      // Draw the piece; p to angle
      l = newl;
      Circle(l) = {p, center, arch_center}; 
      // Let each yet be marked as a different surface
      Physical Line(5+i) = {l};
      cylinder[] += {l};

      // Draw the piece; angle to q
      l = newl;
      Circle(l) = {arch_center, center, q}; 
      // Let each yet be marked as a different surface
      Physical Line(5+i) += {l};
      cylinder[] += {l};
  EndFor

  // Fill in the rest of the cylinder. These are no slip surfaces
  lower_bound[] += {lower_bound[0]};
  Physical Line(4) = {};  // No slip cylinder surfaces
  For i In {0:(n-1)}
    p = upper_bound[i];
    q = lower_bound[i+1];

    pc[] = Point{p}; // Get coordinates
    qc[] = Point{q}; // Get coordinates

    // Compute the angle
    angle_p = Atan2(pc[1], pc[0]);
    angle_p = (angle_p > 0) ? angle_p : (2*Pi + angle_p);

    angle_q = Atan2(qc[1], qc[0]);
    angle_q = (angle_q > 0) ? angle_q : (2*Pi + angle_q);

    angle = angle_q - angle_p; // front back
    angle = (angle < 0) ? angle + 2*Pi : angle; // check also back front

    // Greter than Pi, then we need to insert point
    If(angle > Pi)
      half[] = Rotate {{0, 0, 1}, {0, 0, 0}, angle/2} {Duplicata{Point{p};}};         

      l = newl;
      Circle(l) = {p, center, half}; 
      // Let each yet be marked as a different surface
      Physical Line(4) += {l};
      cylinder[] += {l};

      l = newl;
      Circle(l) = {half, center, q}; 
      // Let each yet be marked as a different surface
      Physical Line(4) += {l};
      cylinder[] += {l};                     
    Else
      l = newl;
      Circle(l) = {p, center, q}; 
      // Let each yet be marked as a different surface
      Physical Line(4) += {l};
      cylinder[] += {l};
    EndIf
  EndFor
// Just the circle
Else
   p = newp; 
   Point(p) = {-jet_radius, 0, 0, cylinder_inner_size};
   Point(p+1) = {0, jet_radius, 0, cylinder_inner_size};
   Point(p+2) = {jet_radius, 0, 0, cylinder_inner_size};
   Point(p+3) = {0, -jet_radius, 0, cylinder_inner_size};
	
   l = newl;
   Circle(l) = {p, center, p+1};
   Circle(l+1) = {p+1, center, p+2};
   Circle(l+2) = {p+2, center, p+3};
   Circle(l+3) = {p+3, center, p};

   cylinder[] = {l, l+1, l+2, l+3};			
   Physical Line(4) = {cylinder[]};
EndIf

// The chanel
A = newp;
Point(A) = {-front_distance, -bottom_distance, 0, inlet_size};

B = newp;
Point(B) = {-front_distance+length, -bottom_distance, 0, outlet_size};

C = newp;
Point(C) = {-front_distance, -bottom_distance+width, 0, inlet_size};

D = newp;
Point(D) = {-front_distance+length, -bottom_distance+width, 0, outlet_size};

AB = newl;
// A no slip wall
Line(AB) = {A, B};
Physical Line(1) = {AB};

// Outflow
BD = newl;
Line(BD) = {B, D};
Physical Line(2) = {BD};

// Top no slip wall
CD = newl;
Line(CD) = {C, D};
Physical Line(1) += {CD};

// Inlet
AC = newl;
Line(AC) = {A, C};
Physical Line(3) = {AC};

// Surface
Line Loop(1) = {AB, BD, -CD, -AC};
Line Loop(2) = {cylinder[]};
Plane Surface(1) = {1, 2};

Physical Surface(1) = {1};

// Control layer close to cylinder
Field[1] = Distance;
Field[1].NodesList = {center};

Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = cylinder_inner_size;
Field[2].LcMax = outlet_size;   
Field[2].DistMin = jet_radius+cylinder_bl_width/2;
Field[2].DistMax = jet_radius+cylinder_bl_width;

// Control wake
wake_width = jet_radius + 2*cylinder_bl_width;

Field[3] = Box;
Field[3].XMin = 0;
Field[3].XMax = jet_radius+wake_length;
Field[3].YMin = -wake_width/2;
Field[3].YMax = wake_width/2;
Field[3].VIn = wake_size;
Field[3].VOut = outlet_size;

// All
Field[5] = Min;
Field[5].FieldsList = {2, 3};

Background Field = 5;     