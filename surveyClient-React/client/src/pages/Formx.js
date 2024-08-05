import React, { useState, useEffect } from "react";
import Form from "react-bootstrap/Form";
import Button from "react-bootstrap/Button";
import Axios from "axios";
import { useNavigate } from "react-router-dom";

function Formx() {
  const [text, setText] = useState("");
  const [response, setResponse] = useState({});

  useEffect(() => {
    const addToList = async () => {
      try {
        const res = await Axios.get("http://127.0.0.1:5000/mine", {
          params: { text: text },
        });

        setResponse(res.data);
      } catch (error) {
        console.error("Error adding to list:", error);
      }
    };
    addToList();
  }, [text]);

  let navigate = useNavigate();

  const handleClick2 = async () => {
    navigate("/nextForm");
  };

  return (
    <div>
      <h1 style={{ fontFamily: "sans-serif", textAlign: "center" }}>
        -Dynamic medical Survey-
      </h1>
      <div
        style={{
          justifyContent: "center",
          alignItems: "center",
          height: "45vh",
          width: "20vh 20vh",
          border: "1px solid gray",
          margin: "70px 70px 70px 70px",
        }}
      >
        <Form>
          <Form.Group className="mb-3" controlId="exampleForm.ControlTextarea1">
            <h3 style={{ fontFamily: "sans-serif", textAlign: "center" }}>
              Ask Question!!
            </h3>

            <Form.Control
              as="textarea"
              rows={7}
              onChange={(e) => {
                setText(e.target.value);
              }}
            />
          </Form.Group>
          <Button
            type="submit"
            size="lg"
            style={{ fontFamily: "sans-serif" }}
            className="w-100"
            onClick={handleClick2}
          >
            Do you want test accurately about your {response.valx}??
          </Button>
        </Form>
      </div>
    </div>
  );
}

export default Formx;
