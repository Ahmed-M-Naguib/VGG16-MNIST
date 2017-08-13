

import imp
import argparse
import logging
from path import Path
# import sys

import numpy as np

import vtk
import numpy as np


def display(input,name):
    def make_data_matrix(x, intensity):
        return intensity * np.repeat(np.repeat(np.repeat(x, v_res, axis=0), v_res, axis=1), v_res, axis=2)

    dataImporter = vtk.vtkImageImport()

    v_res = 3
    dim = 32

    edi = [0]
    dm = [0]

    eIs = [64]
    dm[0] = make_data_matrix(input, eIs[0]).tostring()
    edi[0] = vtk.vtkImageImport()
    edi[0].CopyImportVoidPointer(dm[0], len(dm[0]))
    edi[0].SetDataScalarTypeToUnsignedChar()
    edi[0].SetNumberOfScalarComponents(1)
    edi[0].SetDataExtent(0, int(dim * v_res) - 1, 0, int(dim * v_res) - 1, 0, int(dim * v_res) - 1)
    edi[0].SetWholeExtent(0, int(dim * v_res) - 1, 0, int(dim * v_res) - 1, 0, int(dim * v_res) - 1)
     # Prepare color and transparency values
    colorFunc = vtk.vtkColorTransferFunction()
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    alphaChannelFunc.AddPoint(0, 0.0)
    alphaChannelFunc.AddPoint(64, 1)
    alphaChannelFunc.AddPoint(128, 1.0)
    alphaChannelFunc.AddPoint(192, 1.0)
    alphaChannelFunc.AddPoint(255, 1.0)

    colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)
    colorFunc.AddRGBPoint(64, 0.0, 0.4, 0.8)
    colorFunc.AddRGBPoint(128, 0.8, 0.0, 0.0)
    colorFunc.AddRGBPoint(192, 0.8, 0.0, 0.7)
    colorFunc.AddRGBPoint(255, 0.0, 0.8, 0.0)

    # Prepare volume properties.
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(alphaChannelFunc)
    volumeProperty.ShadeOn()  # Keep this on unless you want everything to look terrible
    volumeProperty.SetInterpolationTypeToNearest()


    # More VTK Bookkeeping stuff.
    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()

    # Specify the data and raycast methods for the rendered volumes.
    volumeMapper = vtk.vtkVolumeRayCastMapper()
    volumeMapper.SetVolumeRayCastFunction(compositeFunction)
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

    # Endpoint volumeMappers
    evm = [0]


    evm[0] = vtk.vtkVolumeRayCastMapper()
    evm[0].SetVolumeRayCastFunction(compositeFunction)
    evm[0].SetInputConnection(edi[0].GetOutputPort())  # Prepare the volume for the draggable model.
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    volume.SetPosition([0, 0, 0])

    # Endpoint volumes
    ev = [0, 0, 0, 0]
    vps = [[-200, -160, -280], [0, 150.0, -150.0], [0, -150.0, 150.0], [0, 150.0, 150.0]]

    ev[0] = vtk.vtkVolume()
    ev[0].SetMapper(evm[0])
    ev[0].SetProperty(volumeProperty)


    ev[0].SetPosition(vps[0])
    ev[0].DragableOff()
    ev[0].PickableOff()


    class MyInteractorStyle(vtk.vtkInteractorStyleSwitch):

        def __init__(self, parent=None):

            # Index indicating which models are currently selected for endpoints
            self.ix = 0

            # Togglable flag indicating if the center model is being dragged or not
            self.drag = 0;

            # Picker
            self.picker = vtk.vtkCellPicker()
            self.picker.SetTolerance(0.001)

    # Initialize the render window
    renderer = vtk.vtkRenderer()
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)

    # Initialize the render interactor
    renderInteractor = vtk.vtkRenderWindowInteractor()
    style = MyInteractorStyle()
    style.SetDefaultRenderer(renderer)
    renderInteractor.SetInteractorStyle(style)  # volume_set=volume,Lvolume_set = Lvolume, Rvolume_set = Rvolume))
    renderInteractor.SetRenderWindow(renderWin)


    renderer.AddVolume(ev[0])
    renderer.AddVolume(volume)

    # set background to white. Optionally change it to a fun color, like "Lifeblood of the Untenderized."
    renderer.SetBackground(1.0, 1.0, 1.0)


    renderWin.SetSize(500, 500)

    # Exit function
    def exitCheck(obj, event):
        if obj.GetEventPending() != 0:
            obj.SetAbortRender(1)

    # Add exit function
    renderWin.AddObserver("AbortCheckEvent", exitCheck)

    # initialize interactor
    renderInteractor.Initialize()

    # Start application!
    renderer.ResetCamera()
    renderer.GetActiveCamera().Azimuth(30)
    renderer.GetActiveCamera().Elevation(20)
    renderer.GetActiveCamera().Dolly(2.8)
    renderer.ResetCameraClippingRange()





    w2i = vtk.vtkWindowToImageFilter()
    writer = vtk.vtkTIFFWriter()
    w2i.SetInput(renderWin)
    w2i.Update()
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.SetFileName("./samples/"+name+".png")


    renderWin.Render()
    # renderInteractor.Start()


    writer.Write()


