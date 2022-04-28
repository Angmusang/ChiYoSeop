from dataclasses import make_dataclass

   Point = make_dataclass("Point", [("x", int), ("y", int)])
   pd.DataFrame([Point(0, 0), Point(0, 3), Point(2, 3)])


