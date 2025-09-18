# trace generated using paraview version 5.11.1
# import paraview
# paraview.compatibility.major = 5
# paraview.compatibility.minor = 11

from pathlib import Path

# import the simple module from the paraview
from paraview.simple import *

# disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

data_path = Path.cwd()
# data_path = (
# Path.home()
# / "projects/aster_invasion/data/inhibition/25-02-14_dc2.2.3.1_inhib1-10log_L280_Lz100_nuc4000-8000_3d_ihr.5-2log/simulations/ihrg2_ihs10_nuc4000"
# )

filenames = sorted(
    list((data_path / "vtk").glob("t_*.vtk")), key=lambda x: int(x.stem.split("_")[1])
)
filenames_str = [str(f) for f in filenames]

LoadPalette(paletteName="BlackBackground")

# create a new 'Legacy VTK Reader'
t_0vtk = LegacyVTKReader(registrationName="t_0.vtk*", FileNames=filenames_str)

# get animation scene
animationScene1 = GetAnimationScene()

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# get active view
renderView1 = GetActiveViewOrCreate("RenderView")

# get the material library
materialLibrary1 = GetMaterialLibrary()

# get display properties
t_0vtkDisplay = GetDisplayProperties(t_0vtk, view=renderView1)

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()


# create a new 'Tube'
tube1 = Tube(registrationName="Tube1", Input=t_0vtk)
tube1.Scalars = [None, ""]
tube1.Vectors = [None, "1"]
tube1.Radius = 1.91000851513

# Properties modified on tube1
tube1.Scalars = ["POINTS", ""]
tube1.Vectors = ["POINTS", "1"]

# show data in view
tube1Display = Show(tube1, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
tube1Display.Representation = "Surface"
tube1Display.ColorArrayName = [None, ""]
tube1Display.SelectTCoordArray = "None"
tube1Display.SelectNormalArray = "TubeNormals"
tube1Display.SelectTangentArray = "None"
tube1Display.OSPRayScaleArray = "TubeNormals"
tube1Display.OSPRayScaleFunction = "PiecewiseFunction"
tube1Display.SelectOrientationVectors = "None"
tube1Display.ScaleFactor = 19.253883212876907
tube1Display.SelectScaleArray = "None"
tube1Display.GlyphType = "Arrow"
tube1Display.GlyphTableIndexArray = "None"
tube1Display.GaussianRadius = 0.9626941606438453
tube1Display.SetScaleArray = ["POINTS", "TubeNormals"]
tube1Display.ScaleTransferFunction = "PiecewiseFunction"
tube1Display.OpacityArray = ["POINTS", "TubeNormals"]
tube1Display.OpacityTransferFunction = "PiecewiseFunction"
tube1Display.DataAxesGrid = "GridAxesRepresentation"
tube1Display.PolarAxes = "PolarAxesRepresentation"
tube1Display.SelectInputVectors = ["POINTS", "TubeNormals"]
tube1Display.WriteLog = ""

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
tube1Display.OSPRayScaleFunction.Points = [
    0.0,
    0.0,
    0.5,
    0.0,
    920.3745727539062,
    0.30000001192092896,
    0.5,
    0.0,
    4095.0,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
tube1Display.ScaleTransferFunction.Points = [
    -0.997456967830658,
    0.0,
    0.5,
    0.0,
    -0.5490886989136623,
    0.30000001192092896,
    0.5,
    0.0,
    0.997456967830658,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
tube1Display.OpacityTransferFunction.Points = [
    -0.997456967830658,
    0.0,
    0.5,
    0.0,
    -0.5490886989136623,
    0.30000001192092896,
    0.5,
    0.0,
    0.997456967830658,
    1.0,
    0.5,
    0.0,
]

# hide data in view
Hide(t_0vtk, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on tube1
tube1.Radius = 0.25

# set scalar coloring
ColorBy(tube1Display, ("CELLS", "Centrosome"))

# rescale color and/or opacity maps used to include current data range
tube1Display.RescaleTransferFunctionToDataRange(True, False)

# get color transfer function/color map for 'Centrosome'
centrosomeLUT = GetColorTransferFunction("Centrosome")

# get opacity transfer function/opacity map for 'Centrosome'
centrosomePWF = GetOpacityTransferFunction("Centrosome")

# get 2D transfer function for 'Centrosome'
centrosomeTF2D = GetTransferFunction2D("Centrosome")

# Apply a preset using its name. Note this may not work as expected when
# presets have duplicate names.
centrosomeLUT.ApplyPreset("PiYG", True)

# hide color bar/color legend
tube1Display.SetScalarBarVisibility(renderView1, False)

# Hide orientation axes
renderView1.OrientationAxesVisibility = 0

renderView1.UseAmbientOcclusion = 1
renderView1.UseToneMapping = 1
renderView1.Exposure = 2.0

renderView1.Update()

# get layout
layout1 = GetLayout()

# split cell
layout1.SplitHorizontal(0, 0.5)
renderView1.Update()


# set active view
SetActiveView(None)

# # get active view
# renderView2 = GetActiveViewOrCreate("RenderView")

# Create a new 'Render View'
renderView2 = CreateView("RenderView")
renderView2.AxesGrid = "GridAxes3DActor"
renderView2.StereoType = "Crystal Eyes"
renderView2.CameraFocalDisk = 1.0
renderView2.BackEnd = "OSPRay raycaster"
renderView2.OSPRayMaterialLibrary = materialLibrary1

# assign view to a particular cell in the layout
AssignViewToLayout(view=renderView2, layout=layout1, hint=2)

# set active source
SetActiveSource(tube1)

# show data in view
tube1Display_1 = Show(tube1, renderView2, "GeometryRepresentation")

# trace defaults for the display properties.
tube1Display_1.Representation = "Surface"
tube1Display_1.ColorArrayName = [None, ""]
tube1Display_1.SelectTCoordArray = "None"
tube1Display_1.SelectNormalArray = "TubeNormals"
tube1Display_1.SelectTangentArray = "None"
tube1Display_1.OSPRayScaleArray = "TubeNormals"
tube1Display_1.OSPRayScaleFunction = "PiecewiseFunction"
tube1Display_1.SelectOrientationVectors = "None"
tube1Display_1.ScaleFactor = 19.12021569788639
tube1Display_1.SelectScaleArray = "None"
tube1Display_1.GlyphType = "Arrow"
tube1Display_1.GlyphTableIndexArray = "None"
tube1Display_1.GaussianRadius = 0.9560107848943196
tube1Display_1.SetScaleArray = ["POINTS", "TubeNormals"]
tube1Display_1.ScaleTransferFunction = "PiecewiseFunction"
tube1Display_1.OpacityArray = ["POINTS", "TubeNormals"]
tube1Display_1.OpacityTransferFunction = "PiecewiseFunction"
tube1Display_1.DataAxesGrid = "GridAxesRepresentation"
tube1Display_1.PolarAxes = "PolarAxesRepresentation"
tube1Display_1.SelectInputVectors = ["POINTS", "TubeNormals"]
tube1Display_1.WriteLog = ""

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
tube1Display_1.OSPRayScaleFunction.Points = [
    0.0,
    0.0,
    0.5,
    0.0,
    920.3745727539062,
    0.30000001192092896,
    0.5,
    0.0,
    4095.0,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
tube1Display_1.ScaleTransferFunction.Points = [
    -0.997456967830658,
    0.0,
    0.5,
    0.0,
    -0.5490886989136623,
    0.30000001192092896,
    0.5,
    0.0,
    0.997456967830658,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
tube1Display_1.OpacityTransferFunction.Points = [
    -0.997456967830658,
    0.0,
    0.5,
    0.0,
    -0.5490886989136623,
    0.30000001192092896,
    0.5,
    0.0,
    0.997456967830658,
    1.0,
    0.5,
    0.0,
]

# reset view to fit data
renderView2.ResetCamera(False)

# update the view to ensure updated data information
renderView2.Update()

# set scalar coloring
ColorBy(tube1Display_1, ("CELLS", "Centrosome"))

# Rescale transfer function
centrosomeLUT.RescaleTransferFunction(-1.5, 1.5)

# Rescale transfer function
centrosomePWF.RescaleTransferFunction(-1.5, 1.5)

# Rescale 2D transfer function
centrosomeTF2D.RescaleTransferFunction(-1.5, 1.5, 0.0, 1.0)

# hide color bar/color legend
tube1Display_1.SetScalarBarVisibility(renderView2, False)

# link cameras in two views
AddCameraLink(renderView2, renderView1, "CameraLink0")

# set active view
SetActiveView(renderView2)

# change interaction mode for render view
renderView2.InteractionMode = "2D"

# create a new 'Clip'
clip1 = Clip(registrationName="Clip1", Input=tube1)
clip1.ClipType = "Plane"
clip1.HyperTreeGridClipper = "Plane"
clip1.Scalars = ["CELLS", "Centrosome"]
clip1.Value = 0.5

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Origin = [0.0, 0.0, 0.5]
clip1.HyperTreeGridClipper.Origin = [0.0, 0.0, 0.5]
clip1.ClipType.Normal = [0.0, 0.0, 1.0]

# show data in view
clip1Display = Show(clip1, renderView2, "UnstructuredGridRepresentation")

# trace defaults for the display properties.
clip1Display.Representation = "Surface"
clip1Display.ColorArrayName = ["CELLS", "Centrosome"]
clip1Display.LookupTable = centrosomeLUT
clip1Display.SelectTCoordArray = "None"
clip1Display.SelectNormalArray = "TubeNormals"
clip1Display.SelectTangentArray = "None"
clip1Display.OSPRayScaleArray = "TubeNormals"
clip1Display.OSPRayScaleFunction = "PiecewiseFunction"
clip1Display.SelectOrientationVectors = "None"
clip1Display.ScaleFactor = 6.210989006180451
clip1Display.SelectScaleArray = "None"
clip1Display.GlyphType = "Arrow"
clip1Display.GlyphTableIndexArray = "None"
clip1Display.GaussianRadius = 0.31054945030902253
clip1Display.SetScaleArray = ["POINTS", "TubeNormals"]
clip1Display.ScaleTransferFunction = "PiecewiseFunction"
clip1Display.OpacityArray = ["POINTS", "TubeNormals"]
clip1Display.OpacityTransferFunction = "PiecewiseFunction"
clip1Display.DataAxesGrid = "GridAxesRepresentation"
clip1Display.PolarAxes = "PolarAxesRepresentation"
clip1Display.ScalarOpacityFunction = centrosomePWF
clip1Display.ScalarOpacityUnitDistance = 6.451098542581446
clip1Display.OpacityArrayName = ["POINTS", "TubeNormals"]
clip1Display.SelectInputVectors = ["POINTS", "TubeNormals"]
clip1Display.WriteLog = ""

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
clip1Display.OSPRayScaleFunction.Points = [
    0.0,
    0.0,
    0.5,
    0.0,
    920.3745727539062,
    0.30000001192092896,
    0.5,
    0.0,
    4095.0,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
clip1Display.ScaleTransferFunction.Points = [
    -0.9856006503105164,
    0.0,
    0.5,
    0.0,
    -0.5425619311722923,
    0.30000001192092896,
    0.5,
    0.0,
    0.9856006503105164,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
clip1Display.OpacityTransferFunction.Points = [
    -0.9856006503105164,
    0.0,
    0.5,
    0.0,
    -0.5425619311722923,
    0.30000001192092896,
    0.5,
    0.0,
    0.9856006503105164,
    1.0,
    0.5,
    0.0,
]

# hide data in view
Hide(tube1, renderView2)

# show color bar/color legend
clip1Display.SetScalarBarVisibility(renderView2, True)

# update the view to ensure updated data information
renderView2.Update()


# reset view to fit data
renderView2.ResetCamera(False)

renderView2.ResetActiveCameraToPositiveZ()

# reset view to fit data
renderView2.ResetCamera(False)

# hide color bar/color legend
clip1Display.SetScalarBarVisibility(renderView2, False)

# create a new 'Clip'
clip2 = Clip(registrationName="Clip2", Input=clip1)
clip2.ClipType = "Plane"
clip2.HyperTreeGridClipper = "Plane"
clip2.Scalars = ["CELLS", "Centrosome"]
clip2.Value = 0.5

# init the 'Plane' selected for 'ClipType'
clip2.ClipType.Origin = [0.0, 0.0, -0.5]
clip2.HyperTreeGridClipper.Origin = [0.0, 0.0, -0.5]
clip2.ClipType.Normal = [0.0, 0.0, 1.0]

# show data in view
clip2Display = Show(clip2, renderView2, "UnstructuredGridRepresentation")

# trace defaults for the display properties.
clip2Display.Representation = "Surface"
clip2Display.ColorArrayName = ["CELLS", "Centrosome"]
clip2Display.LookupTable = centrosomeLUT
clip2Display.SelectTCoordArray = "None"
clip2Display.SelectNormalArray = "TubeNormals"
clip2Display.SelectTangentArray = "None"
clip2Display.OSPRayScaleArray = "TubeNormals"
clip2Display.OSPRayScaleFunction = "PiecewiseFunction"
clip2Display.SelectOrientationVectors = "None"
clip2Display.ScaleFactor = 5.930645203633552
clip2Display.SelectScaleArray = "None"
clip2Display.GlyphType = "Arrow"
clip2Display.GlyphTableIndexArray = "None"
clip2Display.GaussianRadius = 0.29653226018167755
clip2Display.SetScaleArray = ["POINTS", "TubeNormals"]
clip2Display.ScaleTransferFunction = "PiecewiseFunction"
clip2Display.OpacityArray = ["POINTS", "TubeNormals"]
clip2Display.OpacityTransferFunction = "PiecewiseFunction"
clip2Display.DataAxesGrid = "GridAxesRepresentation"
clip2Display.PolarAxes = "PolarAxesRepresentation"
clip2Display.ScalarOpacityFunction = centrosomePWF
clip2Display.ScalarOpacityUnitDistance = 5.714716719422841
clip2Display.OpacityArrayName = ["POINTS", "TubeNormals"]
clip2Display.SelectInputVectors = ["POINTS", "TubeNormals"]
clip2Display.WriteLog = ""

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
clip2Display.OSPRayScaleFunction.Points = [
    0.0,
    0.0,
    0.5,
    0.0,
    920.3745727539062,
    0.30000001192092896,
    0.5,
    0.0,
    4095.0,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
clip2Display.ScaleTransferFunction.Points = [
    -0.9856006503105164,
    0.0,
    0.5,
    0.0,
    -0.5425619311722923,
    0.30000001192092896,
    0.5,
    0.0,
    0.9856006503105164,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
clip2Display.OpacityTransferFunction.Points = [
    -0.9856006503105164,
    0.0,
    0.5,
    0.0,
    -0.5425619311722923,
    0.30000001192092896,
    0.5,
    0.0,
    0.9856006503105164,
    1.0,
    0.5,
    0.0,
]

# hide data in view
Hide(clip1, renderView2)

# update the view to ensure updated data information
renderView2.Update()

# Properties modified on clip2
clip2.Invert = 0

# reset view to fit data
renderView2.ResetCamera(False)

# set active source
SetActiveSource(clip1)

# Properties modified on animationScene1

renderView2.ResetActiveCameraToPositiveZ()

# reset view to fit data
renderView2.ResetCamera(False)

# hide color bar/color legend
clip2Display.SetScalarBarVisibility(renderView2, False)

# Hide orientation axes
renderView2.OrientationAxesVisibility = 0


# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=clip1.ClipType)
HideInteractiveWidgets(proxy=clip2.ClipType)

# set active view
SetActiveView(renderView1)

# set active source
SetActiveSource(tube1)

# create a new 'Annotate Time Filter'
annotateTimeFilter1 = AnnotateTimeFilter(
    registrationName="AnnotateTimeFilter1", Input=tube1
)

# show data in view
annotateTimeFilter1Display = Show(
    annotateTimeFilter1, renderView1, "TextSourceRepresentation"
)


# update the view to ensure updated data information
renderView1.Update()

# Properties modified on annotateTimeFilter1
annotateTimeFilter1.Scale = 0.16666666666
annotateTimeFilter1.Format = "Time: {time:.0f} min"

# update the view to ensure updated data information
renderView1.Update()


# Properties modified on annotateTimeFilter1Display
annotateTimeFilter1Display.FontSize = 99

# set active view
SetActiveView(renderView2)

# set active source
SetActiveSource(clip2)

# update the view to ensure updated data information
renderView2.Update()

# create a new 'Box'
box1 = Box(registrationName="Box1")

# Properties modified on box1
box1.XLength = 50.0
box1.YLength = 4.0

box1.Center = [130.0, -175.0, 0.0]


# show data in view
box1Display2 = Show(box1, renderView2, "GeometryRepresentation")

# trace defaults for the display properties.
box1Display2.Representation = "Surface"
box1Display2.ColorArrayName = [None, ""]
box1Display2.SelectTCoordArray = "TCoords"
box1Display2.SelectNormalArray = "Normals"
box1Display2.SelectTangentArray = "None"
box1Display2.OSPRayScaleArray = "Normals"
box1Display2.OSPRayScaleFunction = "PiecewiseFunction"
box1Display2.SelectOrientationVectors = "None"
box1Display2.ScaleFactor = 0.1
box1Display2.SelectScaleArray = "None"
box1Display2.GlyphType = "Arrow"
box1Display2.GlyphTableIndexArray = "None"
box1Display2.GaussianRadius = 0.005
box1Display2.SetScaleArray = ["POINTS", "Normals"]
box1Display2.ScaleTransferFunction = "PiecewiseFunction"
box1Display2.OpacityArray = ["POINTS", "Normals"]
box1Display2.OpacityTransferFunction = "PiecewiseFunction"
box1Display2.DataAxesGrid = "GridAxesRepresentation"
box1Display2.PolarAxes = "PolarAxesRepresentation"
box1Display2.SelectInputVectors = ["POINTS", "Normals"]
box1Display2.WriteLog = ""

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
box1Display2.OSPRayScaleFunction.Points = [
    0.0,
    0.0,
    0.5,
    0.0,
    920.3745727539062,
    0.30000001192092896,
    0.5,
    0.0,
    4095.0,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
box1Display2.ScaleTransferFunction.Points = [
    -1.0,
    0.0,
    0.5,
    0.0,
    -0.5504886091556014,
    0.30000001192092896,
    0.5,
    0.0,
    1.0,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
box1Display2.OpacityTransferFunction.Points = [
    -1.0,
    0.0,
    0.5,
    0.0,
    -0.5504886091556014,
    0.30000001192092896,
    0.5,
    0.0,
    1.0,
    1.0,
    0.5,
    0.0,
]

# update the view to ensure updated data information
renderView2.Update()


box1Display1 = Show(box1, renderView1, "GeometryRepresentation")
box1Display1.Representation = "Surface"
box1Display1.ColorArrayName = [None, ""]
box1Display1.SelectTCoordArray = "TCoords"
box1Display1.SelectNormalArray = "Normals"
box1Display1.SelectTangentArray = "None"
box1Display1.OSPRayScaleArray = "Normals"
box1Display1.OSPRayScaleFunction = "PiecewiseFunction"
box1Display1.SelectOrientationVectors = "None"
box1Display1.ScaleFactor = 0.1
box1Display1.SelectScaleArray = "None"
box1Display1.GlyphType = "Arrow"
box1Display1.GlyphTableIndexArray = "None"
box1Display1.GaussianRadius = 0.005
box1Display1.SetScaleArray = ["POINTS", "Normals"]
box1Display1.ScaleTransferFunction = "PiecewiseFunction"
box1Display1.OpacityArray = ["POINTS", "Normals"]
box1Display1.OpacityTransferFunction = "PiecewiseFunction"
box1Display1.DataAxesGrid = "GridAxesRepresentation"
box1Display1.PolarAxes = "PolarAxesRepresentation"
box1Display1.SelectInputVectors = ["POINTS", "Normals"]
box1Display1.WriteLog = ""

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
box1Display1.OSPRayScaleFunction.Points = [
    0.0,
    0.0,
    0.5,
    0.0,
    920.3745727539062,
    0.30000001192092896,
    0.5,
    0.0,
    4095.0,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
box1Display1.ScaleTransferFunction.Points = [
    -1.0,
    0.0,
    0.5,
    0.0,
    -0.5504886091556014,
    0.30000001192092896,
    0.5,
    0.0,
    1.0,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
box1Display1.OpacityTransferFunction.Points = [
    -1.0,
    0.0,
    0.5,
    0.0,
    -0.5504886091556014,
    0.30000001192092896,
    0.5,
    0.0,
    1.0,
    1.0,
    0.5,
    0.0,
]

# update the view to ensure updated data information
renderView1.Update()
renderView2.Update()

# create a new '3D Text'
a3DText1 = a3DText(registrationName="50 um")
# Properties modified on a3DText1
a3DText1.Text = "50 um"

# show data in view
a3DText1Display2 = Show(a3DText1, renderView2, "GeometryRepresentation")

# trace defaults for the display properties.
a3DText1Display2.Representation = "Surface"
a3DText1Display2.ColorArrayName = [None, ""]
a3DText1Display2.SelectTCoordArray = "None"
a3DText1Display2.SelectNormalArray = "None"
a3DText1Display2.SelectTangentArray = "None"
a3DText1Display2.OSPRayScaleFunction = "PiecewiseFunction"
a3DText1Display2.SelectOrientationVectors = "None"
a3DText1Display2.ScaleFactor = 0.5701544925570489
a3DText1Display2.SelectScaleArray = "None"
a3DText1Display2.GlyphType = "Arrow"
a3DText1Display2.GlyphTableIndexArray = "None"
a3DText1Display2.GaussianRadius = 0.028507724627852442
a3DText1Display2.SetScaleArray = [None, ""]
a3DText1Display2.ScaleTransferFunction = "PiecewiseFunction"
a3DText1Display2.OpacityArray = [None, ""]
a3DText1Display2.OpacityTransferFunction = "PiecewiseFunction"
a3DText1Display2.DataAxesGrid = "GridAxesRepresentation"
a3DText1Display2.PolarAxes = "PolarAxesRepresentation"
a3DText1Display2.SelectInputVectors = [None, ""]
a3DText1Display2.WriteLog = ""

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
a3DText1Display2.OSPRayScaleFunction.Points = [
    0.0,
    0.0,
    0.5,
    0.0,
    920.3745727539062,
    0.30000001192092896,
    0.5,
    0.0,
    4095.0,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a3DText1Display2.ScaleTransferFunction.Points = [
    0.0,
    0.0,
    0.5,
    0.0,
    0.22475569542219934,
    0.30000001192092896,
    0.5,
    0.0,
    1.0,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a3DText1Display2.OpacityTransferFunction.Points = [
    0.0,
    0.0,
    0.5,
    0.0,
    0.22475569542219934,
    0.30000001192092896,
    0.5,
    0.0,
    1.0,
    1.0,
    0.5,
    0.0,
]

# update the view to ensure updated data information
renderView1.Update()
renderView2.Update()


# Properties modified on a3DText1Display
a3DText1Display2.Orientation = [0.0, 180.0, 0.0]
a3DText1Display2.PolarAxes.Orientation = [0.0, 180.0, 0.0]

# Properties modified on a3DText1Display
a3DText1Display2.Scale = [6.0, 6.0, 6.0]

# Properties modified on a3DText1Display.DataAxesGrid
a3DText1Display2.DataAxesGrid.Scale = [6.0, 6.0, 6.0]

# Properties modified on a3DText1Display.PolarAxes
a3DText1Display2.PolarAxes.Scale = [6.0, 6.0, 6.0]

# Properties modified on a3DText1Display
a3DText1Display2.Origin = [20.7, 34.0, 0.0]

a3DText1Display1 = Show(a3DText1, renderView1, "GeometryRepresentation")

a3DText1Display1.Orientation = [0.0, 180.0, 0.0]
a3DText1Display1.PolarAxes.Orientation = [0.0, 180.0, 0.0]

# Properties modified on a3DText1Display
a3DText1Display1.Scale = [6.0, 6.0, 6.0]

# Properties modified on a3DText1Display.DataAxesGrid
a3DText1Display1.DataAxesGrid.Scale = [6.0, 6.0, 6.0]

# Properties modified on a3DText1Display.PolarAxes
a3DText1Display1.PolarAxes.Scale = [6.0, 6.0, 6.0]

# Properties modified on a3DText1Display
a3DText1Display1.Origin = [20.7, 34.0, 0.0]


# update the view to ensure updated data information
renderView1.Update()
renderView2.Update()
# ================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
# ================================================================

# --------------------------------
# saving layout sizes for layouts

layout1 = GetLayout()
# layout/tab size in pixels
layout1.SetSize(3836, 2160)
width, height = layout1.GetSize()
layout1.SetSize(2 * width, 2 * height)


# Set the layout size to match the chosen resolution
# layout1.SetSize(width, height)

# Enter preview mode
layout1.PreviewMode = [2 * width, 2 * height]

# -----------------------------------
# saving camera placements for views

# RemoveCameraLink("CameraLink0")

# current camera placement for renderView1
renderView1.InteractionMode = "3D"
renderView1.CameraPosition = [
    0.0,
    0.0,
    -690.0,
]
renderView1.CameraFocalPoint = [
    0.0,
    0.0,
    -20.0,
]
renderView1.CameraParallelScale = 161.39859520904727
renderView1.Update()

# # current camera placement for renderView2_1
# renderView2.InteractionMode = "2D"
# renderView2.CameraPosition = [
#     0.004528583436183453,
#     -0.0033397597286466407,
#     -1114.0331048863113,
# ]
# renderView2.CameraFocalPoint = [
#     0.004528583436183453,
#     -0.0033397597286466407,
#     -20.12486679968056,
# ]
# renderView2.CameraParallelScale = 161.39859520904727
renderView2.Update()


print(width, height)
# --------------------------------------------
# uncomment the following to render all views
RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
# save animation
SaveAnimation(
    str(data_path / (data_path.name + ".avi")),
    layout1,
    SaveAllViews=1,
    ImageResolution=[width, height],
    FontScaling="Do not scale fonts",
    FrameRate=15,
    FrameWindow=[0, len(filenames) - 1],
    # FrameWindow=[30, 50],
)

# SaveScreenshot(
#     str(data_path / (data_path.stem + ".png")),
#     layout1,
#     SaveAllViews=1,
#     ImageResolution=[width, height],
#     FontScaling="Do not scale fonts",
# )
