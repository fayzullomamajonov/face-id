function openCamera() {
    var cameraFeed = document.getElementById('camera_feed');
    
    // Access the laptop camera
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            cameraFeed.srcObject = stream;
            cameraFeed.style.display = 'block';
            
            // After 3 seconds, stop the video stream and redirect to the specified URL
            setTimeout(function() {
                cameraFeed.pause();
                cameraFeed.srcObject.getTracks().forEach(function(track) {
                    track.stop();
                });
                cameraFeed.style.display = 'none';
                
            }, 2000);
            window.location.href = "/face_recognition/";
        })
        .catch(function(error) {
            console.error('Error accessing the camera: ', error);
        });
}