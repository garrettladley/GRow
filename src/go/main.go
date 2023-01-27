package main

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

func main() {
	router := gin.Default()
	router.GET("/test", test)

	router.Run("localhost:4000")
}

func test(c *gin.Context) {
	// cmd := exec.Command("python", "python/script.py")
	// cmd.Dir = "../"
	// output, err := cmd.Output()
	// if err != nil {
	// 	c.String(500, err.Error())
	// 	return
	// }
	// c.String(200, string(output))
	c.String(http.StatusOK, "hello!")
}
