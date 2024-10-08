import React, { useRef, useState, useEffect, useCallback } from 'react';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import '@tensorflow/tfjs';
import { Card, CardContent, Typography, TextField, Button, Box } from '@mui/material';

const ObjectDetection = () => {
  const fileInputRef = useRef(null);
  const [imageName, setImageName] = useState('');
  const [imageSrc, setImageSrc] = useState('');
  const [predictions, setPredictions] = useState([]);
  const canvasRef = useRef(null);
  const modelRef = useRef(null);

  const handleFileUpload = useCallback(async (event) => {
    const file = event.target.files[0];
    if (file) {
      setImageName(file.name);

      const reader = new FileReader();
      reader.onload = (e) => {
        const imageDataURL = e.target.result;
        setImageSrc(imageDataURL);
      };

      reader.readAsDataURL(file);
    }
  }, []);

  const detectObjects = useCallback(async () => {
    if (!modelRef.current || !imageSrc) return;

    const imageElement = new Image();
    imageElement.src = imageSrc;

    imageElement.onload = async () => {
      const predictions = await modelRef.current.detect(imageElement);
      setPredictions(predictions);
    };
  }, [imageSrc]);

  const drawBoundingBoxes = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!canvas || !imageSrc) {
      console.error('Canvas or image source not available.');
      return;
    }

    const imageElement = new Image();
    imageElement.src = imageSrc;

    imageElement.onload = () => {
      canvas.width = imageElement.width;
      canvas.height = imageElement.height;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);

      predictions.forEach((prediction) => {
        const [x, y, width, height] = prediction.bbox;

        ctx.beginPath();
        ctx.rect(x, y, width, height);
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'red';
        ctx.fillStyle = 'transparent';
        ctx.stroke();

        ctx.font = '14px Arial';
        ctx.fillStyle = 'red';
        ctx.fillText(
          `${prediction.class} - ${Math.round(prediction.score * 100)}%`,
          x,
          y > 10 ? y - 5 : 10
        );
      });
    };
  }, [imageSrc, predictions]);

  useEffect(() => {
    (async () => {
      modelRef.current = await cocoSsd.load();
    })();
  }, []);

  useEffect(() => {
    detectObjects();
  }, [imageSrc, detectObjects]);

  useEffect(() => {
    if (imageSrc && predictions.length > 0) {
      drawBoundingBoxes();
    }
  }, [imageSrc, predictions, drawBoundingBoxes]);

  return (
    <Box
      paddingTop={5}
      display="flex"
      flexDirection="column"
      alignItems="center"
      justifyContent="center"
      height="100vh"
    >
      <TextField
        type="file"
        InputLabelProps={{
          shrink: true,
        }}
        variant="outlined"
        margin="normal"
        fullWidth
        InputProps={{
          inputProps: {
            accept: 'image/*',
          },
        }}
        onChange={handleFileUpload}
        inputRef={fileInputRef}
        style={{ width: '300px' }}
      />

      <Button variant="contained" color="primary" onClick={detectObjects}>
        Detect Objects
      </Button>

      {imageName && (
        <Box display="flex" justifyContent="center" marginTop="20px">
          <Card style={{ maxWidth: '600px', marginRight: '20px' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Image Preview
              </Typography>
              <img
                src={imageSrc}
                alt="Uploaded"
                style={{
                  width: '100%',
                  height: 'auto',
                  maxHeight: '400px',
                }}
              />
              <Typography variant="subtitle1" style={{ marginTop: 10 }}>
                Image Name: {imageName}
              </Typography>
            </CardContent>
          </Card>

          <Card style={{ border: '1px solid #ddd', maxWidth: '600px' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Object Detection
              </Typography>
              <canvas
                ref={canvasRef}
                style={{ marginTop: 20, width: '100%', border: '1px solid #ddd' }}
              />
            </CardContent>
          </Card>
        </Box>
      )}
    </Box>
  );
};

export default ObjectDetection;
