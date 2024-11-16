async function predict() {
    // Get values from input fields
    const bedrooms = document.getElementById("bedrooms").value;
    const bathrooms = document.getElementById("bathrooms").value;
    const floors = document.getElementById("floors").value;
    const year = document.getElementById("year").value;

    // Check if all fields are filled out
    if (!bedrooms || !bathrooms || !floors || !year) {
        alert("Please enter all values!");
        return;
    }

    try {
        // Send POST request to the Flask backend
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                bedrooms: parseInt(bedrooms),
                bathrooms: parseInt(bathrooms),
                floors: parseInt(floors),
                year: parseInt(year)
            })
        });

        // Handle the response
        if (response.ok) {
            const data = await response.json();
            document.getElementById("result").innerText = `$${data.prediction.toFixed(2)}`;
            document.getElementById("result-container").style.display = "block";
        } else {
            alert("Prediction failed. Please try again.");
        }
    } catch (error) {
        console.error("Error connecting to the API:", error);
        alert("Cannot connect to the server. Make sure the server is running.");
    }
}
